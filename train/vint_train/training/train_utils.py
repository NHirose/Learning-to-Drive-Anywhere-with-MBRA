import wandb
import os
import numpy as np
import yaml
from typing import List, Optional, Dict
from prettytable import PrettyTable
import tqdm
import itertools

from vint_train.visualizing.action_utils import visualize_traj_pred, plot_trajs_and_points
from vint_train.visualizing.distance_utils import visualize_dist_pred
from vint_train.visualizing.visualize_utils import to_numpy, from_numpy
from vint_train.training.logger import Logger
from vint_train.data.data_utils import VISUALIZATION_IMAGE_SIZE
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import transforms
from torchvision.utils import save_image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import clip
import pickle
import cv2
import random
from PIL import Image
#from torch.cuda.amp import autocast

import psutil
import copy

# LOAD DATA CONFIG
with open(os.path.join(os.path.dirname(__file__), "../data/data_config.yaml"), "r") as f:
    data_config = yaml.safe_load(f)
# POPULATE ACTION STATS
ACTION_STATS = {}
for key in data_config['action_stats']:
    ACTION_STATS[key] = np.array(data_config['action_stats'][key])

def get_current_lr(optimizer):
    return [param_group['lr'] for param_group in optimizer.param_groups]
    
# Train utils for ViNT and GNM
def _compute_losses(
    dist_label: torch.Tensor,
    action_label: torch.Tensor,
    dist_pred: torch.Tensor,
    action_pred: torch.Tensor,
    alpha: float,
    learn_angle: bool,
    action_mask: torch.Tensor = None,
):
    """
    Compute losses for distance and action prediction.

    """
    dist_loss = F.mse_loss(dist_pred.squeeze(-1), dist_label.float())

    def action_reduce(unreduced_loss: torch.Tensor):
        # Reduce over non-batch dimensions to get loss per batch element
        while unreduced_loss.dim() > 1:
            unreduced_loss = unreduced_loss.mean(dim=-1)
        assert unreduced_loss.shape == action_mask.shape, f"{unreduced_loss.shape} != {action_mask.shape}"
        return (unreduced_loss * action_mask).mean() / (action_mask.mean() + 1e-2)

    # Mask out invalid inputs (for negatives, or when the distance between obs and goal is large)
    assert action_pred.shape == action_label.shape, f"{action_pred.shape} != {action_label.shape}"
    action_loss = action_reduce(F.mse_loss(action_pred, action_label, reduction="none"))

    action_waypts_cos_similairity = action_reduce(F.cosine_similarity(
        action_pred[:, :, :2], action_label[:, :, :2], dim=-1
    ))
    multi_action_waypts_cos_sim = action_reduce(F.cosine_similarity(
        torch.flatten(action_pred[:, :, :2], start_dim=1),
        torch.flatten(action_label[:, :, :2], start_dim=1),
        dim=-1,
    ))

    results = {
        "dist_loss": dist_loss,
        "action_loss": action_loss,
        "action_waypts_cos_sim": action_waypts_cos_similairity,
        "multi_action_waypts_cos_sim": multi_action_waypts_cos_sim,
    }

    if learn_angle:
        action_orien_cos_sim = action_reduce(F.cosine_similarity(
            action_pred[:, :, 2:], action_label[:, :, 2:], dim=-1
        ))
        multi_action_orien_cos_sim = action_reduce(F.cosine_similarity(
            torch.flatten(action_pred[:, :, 2:], start_dim=1),
            torch.flatten(action_label[:, :, 2:], start_dim=1),
            dim=-1,
            )
        )
        results["action_orien_cos_sim"] = action_orien_cos_sim
        results["multi_action_orien_cos_sim"] = multi_action_orien_cos_sim

    total_loss = alpha * 1e-2 * dist_loss + (1 - alpha) * action_loss
    results["total_loss"] = total_loss

    return results

def _compute_losses_gps(
    action_label: torch.Tensor,
    action_pred: torch.Tensor,
    learn_angle: bool,
    action_mask: torch.Tensor = None,
):
    """
    Compute losses for distance and action prediction.

    """
    def action_reduce(unreduced_loss: torch.Tensor):
        # Reduce over non-batch dimensions to get loss per batch element
        while unreduced_loss.dim() > 1:
            unreduced_loss = unreduced_loss.mean(dim=-1)
        assert unreduced_loss.shape == action_mask.shape, f"{unreduced_loss.shape} != {action_mask.shape}"
        return (unreduced_loss * action_mask).mean() / (action_mask.mean() + 1e-2)

    # Mask out invalid inputs (for negatives, or when the distance between obs and goal is large)
    assert action_pred.shape == action_label.shape, f"{action_pred.shape} != {action_label.shape}"
    action_loss = F.mse_loss(action_pred, action_label, reduction="mean")


    action_waypts_cos_similairity = action_reduce(F.cosine_similarity(
        action_pred[:, :, :2], action_label[:, :, :2], dim=-1
    ))
    multi_action_waypts_cos_sim = action_reduce(F.cosine_similarity(
        torch.flatten(action_pred[:, :, :2], start_dim=1),
        torch.flatten(action_label[:, :, :2], start_dim=1),
        dim=-1,
    ))

    results = {
        "action_loss": action_loss,
        "action_waypts_cos_sim": action_waypts_cos_similairity,
        "multi_action_waypts_cos_sim": multi_action_waypts_cos_sim,
    }

    if learn_angle:
        action_orien_cos_sim = action_reduce(F.cosine_similarity(
            action_pred[:, :, 2:], action_label[:, :, 2:], dim=-1
        ))
        multi_action_orien_cos_sim = action_reduce(F.cosine_similarity(
            torch.flatten(action_pred[:, :, 2:], start_dim=1),
            torch.flatten(action_label[:, :, 2:], start_dim=1),
            dim=-1,
            )
        )
        results["action_orien_cos_sim"] = action_orien_cos_sim
        results["multi_action_orien_cos_sim"] = multi_action_orien_cos_sim

    total_loss = action_loss
    results["total_loss"] = total_loss

    return results

def geometry_criterion_weight(pc, rsize, step_size, weight_geo, device):
    #Input:
    #    pc: estimated point cloud on the virtual robto coordinate,  batch size x step_size x 3 x 128 x 416
    #    rsize: randomized robot radius, batch size x 1
    #    step_size: the number of the virtual control step (control horizon)
    #    device: device id (CPU or GPU)
    #
    #Output:
    #    average of the geometric loss
    
    pred_clamp = []
    pred_ref = []
    MSE = 0
    bias = 10
    bs, seq, nch, hs, ws = pc.size()
    #print(pc.size())

    pred = torch.sqrt(pc[:,:,0]**2 + pc[:,:,2]**2) 
    yaxis = pc[:,:,1]
    
    pc_1 = torch.cat((pc[:,:,:,0:1,:],pc[:,:,:,0:127,:]), axis=3)
    pc_2 = torch.cat((pc[:,:,:,1:128,:],pc[:,:,:,127:128,:]), axis=3)
    pc_3 = torch.cat((pc[:,:,:,:,0:1],pc[:,:,:,:,0:415]), axis=4)
    pc_4 = torch.cat((pc[:,:,:,:,1:416],pc[:,:,:,:,415:416]), axis=4)
    weight = (torch.sqrt(torch.sum(torch.square(pc_1 - pc_2), 2))) * (torch.sqrt(torch.sum(torch.square(pc_3 - pc_4), 2)))
    weight = (weight)[:,:,:,bias:416-bias]

    count = 0
    for i in range(bs):
        mask1 = (yaxis[i:i+1,:,:] < 0.15*torch.ones((1, step_size, 128, 416), device=device))
        mask2 = (yaxis[i:i+1,:,:] > -0.3*torch.ones((1, step_size, 128, 416), device=device))
            
        mask = torch.logical_and(mask1, mask2)[:,:,:,bias:416-bias]

        pred_cap = torch.clamp(pred[i:i+1,:,:], 0.0, rsize[i, 0].item())[:,:,:,bias:416-bias]
        pred_cap_mask = pred_cap[mask]
        weight_mask = weight[i:i+1][mask]
        weight_mask = torch.clamp(weight_mask, 0.0, 0.01)

        num_masked = torch.sum((pred_cap_mask == rsize[i, 0].item()).float())
            
        num_m = list(pred_cap_mask.size())[0]
        count = num_m - num_masked.cpu().float().item()
        pred_ref = rsize[i, 0].item()*torch.ones(num_m).to(device)

        MSE += weight_geo[i]*torch.sum(weight_mask*(pred_cap_mask - pred_ref)**2)/(count + 1e-7)*2.0e+3

    return MSE/bs

def geometry_criterion(pc, rsize, step_size, device):
    #Input:
    #    pc: estimated point cloud on the virtual robto coordinate,  batch size x step_size x 3 x 128 x 416
    #    rsize: randomized robot radius, batch size x 1
    #    step_size: the number of the virtual control step (control horizon)
    #    device: device id (CPU or GPU)
    #
    #Output:
    #    average of the geometric loss
    
    pred_clamp = []
    pred_ref = []
    MSE = 0
    bias = 10
    bs, seq, nch, hs, ws = pc.size()

    pred = torch.sqrt(pc[:,:,0]**2 + pc[:,:,2]**2) 
    yaxis = pc[:,:,1]
    
    pc_1 = torch.cat((pc[:,:,:,0:1,:],pc[:,:,:,0:127,:]), axis=3)
    pc_2 = torch.cat((pc[:,:,:,1:128,:],pc[:,:,:,127:128,:]), axis=3)
    pc_3 = torch.cat((pc[:,:,:,:,0:1],pc[:,:,:,:,0:415]), axis=4)
    pc_4 = torch.cat((pc[:,:,:,:,1:416],pc[:,:,:,:,415:416]), axis=4)
    weight = (torch.sqrt(torch.sum(torch.square(pc_1 - pc_2), 2))) * (torch.sqrt(torch.sum(torch.square(pc_3 - pc_4), 2)))
    weight = (weight)[:,:,:,bias:416-bias]

    count = 0
    for i in range(bs):
        mask1 = (yaxis[i:i+1,:,:] < 0.0*torch.ones((1, step_size, 128, 416), device=device))
        mask2 = (yaxis[i:i+1,:,:] > -0.3*torch.ones((1, step_size, 128, 416), device=device))
            
        mask = torch.logical_and(mask1, mask2)[:,:,:,bias:416-bias]

        pred_cap = torch.clamp(pred[i:i+1,:,:], 0.0, rsize[i, 0].item())[:,:,:,bias:416-bias]
        pred_cap_mask = pred_cap[mask]
        weight_mask = weight[i:i+1][mask]
        weight_mask = torch.clamp(weight_mask, 0.0, 0.01)

        num_masked = torch.sum((pred_cap_mask == rsize[i, 0].item()).float())
            
        num_m = list(pred_cap_mask.size())[0]
        count = num_m - num_masked.cpu().float().item()
        pred_ref = rsize[i, 0].item()*torch.ones(num_m).to(device)

        MSE += torch.sum(weight_mask*(pred_cap_mask - pred_ref)**2)/(count + 1e-7)*2.0e+3

    return MSE/bs

def geometry_criterion_range(pc, rsize, step_size, limit_range, device):
    #Input:
    #    pc: estimated point cloud on the virtual robto coordinate,  batch size x step_size x 3 x 128 x 416
    #    rsize: randomized robot radius, batch size x 1
    #    step_size: the number of the virtual control step (control horizon)
    #    device: device id (CPU or GPU)
    #
    #Output:
    #    average of the geometric loss
    
    pred_clamp = []
    pred_ref = []
    MSE = 0
    #bias = 10
    bias = 20
    bs, seq, nch, hs, ws = pc.size()
    #print(pc.size())

    pred = torch.sqrt(pc[:,:,0]**2 + pc[:,:,2]**2) 
    yaxis = pc[:,:,1]
    
    pc_1 = torch.cat((pc[:,:,:,0:1,:],pc[:,:,:,0:127,:]), axis=3)
    pc_2 = torch.cat((pc[:,:,:,1:128,:],pc[:,:,:,127:128,:]), axis=3)
    pc_3 = torch.cat((pc[:,:,:,:,0:1],pc[:,:,:,:,0:415]), axis=4)
    pc_4 = torch.cat((pc[:,:,:,:,1:416],pc[:,:,:,:,415:416]), axis=4)
    weight = (torch.sqrt(torch.sum(torch.square(pc_1 - pc_2), 2))) * (torch.sqrt(torch.sum(torch.square(pc_3 - pc_4), 2)))
    weight = (weight)[:,:,:,bias:416-bias]

    count = 0
    for i in range(bs):
        mask1 = (yaxis[i:i+1,:,:] < limit_range[i,0]*torch.ones((1, step_size, 128, 416), device=device))
        mask2 = (yaxis[i:i+1,:,:] > limit_range[i,1]*torch.ones((1, step_size, 128, 416), device=device))
            
        mask = torch.logical_and(mask1, mask2)[:,:,:,bias:416-bias]

        pred_cap = torch.clamp(pred[i:i+1,:,:], 0.0, rsize[i, 0].item())[:,:,:,bias:416-bias]
        pred_cap_mask = pred_cap[mask]
        weight_mask = weight[i:i+1][mask]
        weight_mask = torch.clamp(weight_mask, 0.0, 0.01)

        num_masked = torch.sum((pred_cap_mask == rsize[i, 0].item()).float())
            
        num_m = list(pred_cap_mask.size())[0]
        count = num_m - num_masked.cpu().float().item()
        pred_ref = rsize[i, 0].item()*torch.ones(num_m).to(device)

        MSE += torch.sum(weight_mask*(pred_cap_mask - pred_ref)**2)/(count + 1e-7)*2.0e+3

    return MSE/bs

def _log_data(
    i,
    epoch,
    num_batches,
    normalized,
    project_folder,
    num_images_log,
    loggers,
    obs_image,
    goal_image,
    action_pred,
    action_label,
    dist_pred,
    dist_label,
    goal_pos,
    dataset_index,
    use_wandb,
    mode,
    use_latest,
    wandb_log_freq=1,
    print_log_freq=1,
    image_log_freq=1,
    wandb_increment_step=True,
):
    """
    Log data to wandb and print to console.
    """
    data_log = {}
    for key, logger in loggers.items():
        if use_latest:
            data_log[logger.full_name()] = logger.latest()
            if i % print_log_freq == 0 and print_log_freq != 0:
                print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")
        else:
            data_log[logger.full_name()] = logger.average()
            if i % print_log_freq == 0 and print_log_freq != 0:
                print(f"(epoch {epoch}) {logger.full_name()} {logger.average()}")

    if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
        wandb.log(data_log, commit=wandb_increment_step)

    if image_log_freq != 0 and i % image_log_freq == 0:
        visualize_dist_pred(
            to_numpy(obs_image),
            to_numpy(goal_image),
            to_numpy(dist_pred),
            to_numpy(dist_label),
            mode,
            project_folder,
            epoch,
            num_images_log,
            use_wandb=use_wandb,
        )
        visualize_traj_pred(
            to_numpy(obs_image),
            to_numpy(goal_image),
            to_numpy(dataset_index),
            to_numpy(goal_pos),
            to_numpy(action_pred),
            to_numpy(action_label),
            mode,
            normalized,
            project_folder,
            epoch,
            num_images_log,
            use_wandb=use_wandb,
        )


def train(
    model: nn.Module,
    optimizer: Adam,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    project_folder: str,
    normalized: bool,
    epoch: int,
    alpha: float = 0.5,
    learn_angle: bool = True,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    use_wandb: bool = True,
    use_tqdm: bool = True,
):
    """
    Train the model for one epoch.

    Args:
        model: model to train
        optimizer: optimizer to use
        dataloader: dataloader for training
        transform: transform to use
        device: device to use
        project_folder: folder to save images to
        epoch: current epoch
        alpha: weight of action loss
        learn_angle: whether to learn the angle of the action
        print_log_freq: how often to print loss
        image_log_freq: how often to log images
        num_images_log: number of images to log
        use_wandb: whether to use wandb
        use_tqdm: whether to use tqdm
    """
    model.train()
    dist_loss_logger = Logger("dist_loss", "train", window_size=print_log_freq)
    action_loss_logger = Logger("action_loss", "train", window_size=print_log_freq)
    action_waypts_cos_sim_logger = Logger(
        "action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    multi_action_waypts_cos_sim_logger = Logger(
        "multi_action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    total_loss_logger = Logger("total_loss", "train", window_size=print_log_freq)
    loggers = {
        "dist_loss": dist_loss_logger,
        "action_loss": action_loss_logger,
        "action_waypts_cos_sim": action_waypts_cos_sim_logger,
        "multi_action_waypts_cos_sim": multi_action_waypts_cos_sim_logger,
        "total_loss": total_loss_logger,
    }

    if learn_angle:
        action_orien_cos_sim_logger = Logger(
            "action_orien_cos_sim", "train", window_size=print_log_freq
        )
        multi_action_orien_cos_sim_logger = Logger(
            "multi_action_orien_cos_sim", "train", window_size=print_log_freq
        )
        loggers["action_orien_cos_sim"] = action_orien_cos_sim_logger
        loggers["multi_action_orien_cos_sim"] = multi_action_orien_cos_sim_logger

    num_batches = len(dataloader)
    tqdm_iter = tqdm.tqdm(
        dataloader,
        disable=not use_tqdm,
        dynamic_ncols=True,
        desc=f"Training epoch {epoch}",
    )
    for i, data in enumerate(tqdm_iter):
        (
            obs_image,
            goal_image,
            action_label,
            dist_label,
            goal_pos,
            dataset_index,
            action_mask,
        ) = data

        obs_images = torch.split(obs_image, 3, dim=1)
        viz_obs_image = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE)
        obs_images = [transform(obs_image).to(device) for obs_image in obs_images]
        obs_image = torch.cat(obs_images, dim=1)

        viz_goal_image = TF.resize(goal_image, VISUALIZATION_IMAGE_SIZE)
        
        goal_image = transform(goal_image).to(device)
        model_outputs = model(obs_image, goal_image)

        dist_label = dist_label.to(device)
        action_label = action_label.to(device)
        action_mask = action_mask.to(device)

        optimizer.zero_grad()
      
        dist_pred, action_pred = model_outputs

        losses = _compute_losses(
            dist_label=dist_label,
            action_label=action_label,
            dist_pred=dist_pred,
            action_pred=action_pred,
            alpha=alpha,
            learn_angle=learn_angle,
            action_mask=action_mask,
        )

        losses["total_loss"].backward()
        optimizer.step()

        for key, value in losses.items():
            if key in loggers:
                logger = loggers[key]
                logger.log_data(value.item())

        _log_data(
            i=i,
            epoch=epoch,
            num_batches=num_batches,
            normalized=normalized,
            project_folder=project_folder,
            num_images_log=num_images_log,
            loggers=loggers,
            obs_image=viz_obs_image,
            goal_image=viz_goal_image,
            action_pred=action_pred,
            action_label=action_label,
            dist_pred=dist_pred,
            dist_label=dist_label,
            goal_pos=goal_pos,
            dataset_index=dataset_index,
            wandb_log_freq=wandb_log_freq,
            print_log_freq=print_log_freq,
            image_log_freq=image_log_freq,
            use_wandb=use_wandb,
            mode="train",
            use_latest=True,
        )


def evaluate(
    eval_type: str,
    model: nn.Module,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    project_folder: str,
    normalized: bool,
    epoch: int = 0,
    alpha: float = 0.5,
    learn_angle: bool = True,
    num_images_log: int = 8,
    use_wandb: bool = True,
    eval_fraction: float = 1.0,
    use_tqdm: bool = True,

):
    """
    Evaluate the model on the given evaluation dataset.

    Args:
        eval_type (string): f"{data_type}_{eval_type}" (e.g. "recon_train", "gs_test", etc.)
        model (nn.Module): model to evaluate
        dataloader (DataLoader): dataloader for eval
        transform (transforms): transform to apply to images
        device (torch.device): device to use for evaluation
        project_folder (string): path to project folder
        epoch (int): current epoch
        alpha (float): weight for action loss
        learn_angle (bool): whether to learn the angle of the action
        num_images_log (int): number of images to log
        use_wandb (bool): whether to use wandb for logging
        eval_fraction (float): fraction of data to use for evaluation
        use_tqdm (bool): whether to use tqdm for logging
    """
    model.eval()
    dist_loss_logger = Logger("dist_loss", eval_type)
    action_loss_logger = Logger("action_loss", eval_type)
    action_waypts_cos_sim_logger = Logger("action_waypts_cos_sim", eval_type)
    multi_action_waypts_cos_sim_logger = Logger("multi_action_waypts_cos_sim", eval_type)
    total_loss_logger = Logger("total_loss", eval_type)
    loggers = {
        "dist_loss": dist_loss_logger,
        "action_loss": action_loss_logger,
        "action_waypts_cos_sim": action_waypts_cos_sim_logger,
        "multi_action_waypts_cos_sim": multi_action_waypts_cos_sim_logger,
        "total_loss": total_loss_logger,
    }

    if learn_angle:
        action_orien_cos_sim_logger = Logger("action_orien_cos_sim", eval_type)
        multi_action_orien_cos_sim_logger = Logger("multi_action_orien_cos_sim", eval_type)
        loggers["action_orien_cos_sim"] = action_orien_cos_sim_logger
        loggers["multi_action_orien_cos_sim"] = multi_action_orien_cos_sim_logger

    num_batches = len(dataloader)
    num_batches = max(int(num_batches * eval_fraction), 1)

    viz_obs_image = None
    with torch.no_grad():
        tqdm_iter = tqdm.tqdm(
            itertools.islice(dataloader, num_batches),
            total=num_batches,
            disable=not use_tqdm,
            dynamic_ncols=True,
            desc=f"Evaluating {eval_type} for epoch {epoch}",
        )
        for i, data in enumerate(tqdm_iter):
            (
                obs_image,
                goal_image,
                action_label,
                dist_label,
                goal_pos,
                dataset_index,
                action_mask,
            ) = data

            obs_images = torch.split(obs_image, 3, dim=1)
            viz_obs_image = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE)
            obs_images = [transform(obs_image).to(device) for obs_image in obs_images]
            obs_image = torch.cat(obs_images, dim=1)

            viz_goal_image = TF.resize(goal_image, VISUALIZATION_IMAGE_SIZE)

            goal_image = transform(goal_image).to(device)
            model_outputs = model(obs_image, goal_image)

            dist_label = dist_label.to(device)
            action_label = action_label.to(device)
            action_mask = action_mask.to(device)

            dist_pred, action_pred = model_outputs

            losses = _compute_losses(
                dist_label=dist_label,
                action_label=action_label,
                dist_pred=dist_pred,
                action_pred=action_pred,
                alpha=alpha,
                learn_angle=learn_angle,
                action_mask=action_mask,
            )

            for key, value in losses.items():
                if key in loggers:
                    logger = loggers[key]
                    logger.log_data(value.item())

    # Log data to wandb/console, with visualizations selected from the last batch
    _log_data(
        i=i,
        epoch=epoch,
        num_batches=num_batches,
        normalized=normalized,
        project_folder=project_folder,
        num_images_log=num_images_log,
        loggers=loggers,
        obs_image=viz_obs_image,
        goal_image=viz_goal_image,
        action_pred=action_pred,
        action_label=action_label,
        goal_pos=goal_pos,
        dist_pred=dist_pred,
        dist_label=dist_label,
        dataset_index=dataset_index,
        use_wandb=use_wandb,
        mode=eval_type,
        use_latest=False,
        wandb_increment_step=False,
    )

    return dist_loss_logger.average(), action_loss_logger.average(), total_loss_logger.average()


# Train utils for NOMAD

def _compute_losses_nomad(
    ema_model,
    noise_scheduler,
    batch_obs_images,
    batch_goal_images,
    batch_dist_label: torch.Tensor,
    batch_action_label: torch.Tensor,
    device: torch.device,
    action_mask: torch.Tensor,
):
    """
    Compute losses for distance and action prediction.
    """

    pred_horizon = batch_action_label.shape[1]
    action_dim = batch_action_label.shape[2]

    model_output_dict = model_output(
        ema_model,
        noise_scheduler,
        batch_obs_images,
        batch_goal_images,
        pred_horizon,
        action_dim,
        num_samples=1,
        device=device,
    )
    uc_actions = model_output_dict['uc_actions']
    gc_actions = model_output_dict['gc_actions']
    gc_distance = model_output_dict['gc_distance']

    gc_dist_loss = F.mse_loss(gc_distance, batch_dist_label.unsqueeze(-1))

    def action_reduce(unreduced_loss: torch.Tensor):
        # Reduce over non-batch dimensions to get loss per batch element
        while unreduced_loss.dim() > 1:
            unreduced_loss = unreduced_loss.mean(dim=-1)
        assert unreduced_loss.shape == action_mask.shape, f"{unreduced_loss.shape} != {action_mask.shape}"
        return (unreduced_loss * action_mask).mean() / (action_mask.mean() + 1e-2)

    # Mask out invalid inputs (for negatives, or when the distance between obs and goal is large)
    assert uc_actions.shape == batch_action_label.shape, f"{uc_actions.shape} != {batch_action_label.shape}"
    assert gc_actions.shape == batch_action_label.shape, f"{gc_actions.shape} != {batch_action_label.shape}"

    uc_action_loss = action_reduce(F.mse_loss(uc_actions, batch_action_label, reduction="none"))
    gc_action_loss = action_reduce(F.mse_loss(gc_actions, batch_action_label, reduction="none"))

    uc_action_waypts_cos_similairity = action_reduce(F.cosine_similarity(
        uc_actions[:, :, :2], batch_action_label[:, :, :2], dim=-1
    ))
    uc_multi_action_waypts_cos_sim = action_reduce(F.cosine_similarity(
        torch.flatten(uc_actions[:, :, :2], start_dim=1),
        torch.flatten(batch_action_label[:, :, :2], start_dim=1),
        dim=-1,
    ))

    gc_action_waypts_cos_similairity = action_reduce(F.cosine_similarity(
        gc_actions[:, :, :2], batch_action_label[:, :, :2], dim=-1
    ))
    gc_multi_action_waypts_cos_sim = action_reduce(F.cosine_similarity(
        torch.flatten(gc_actions[:, :, :2], start_dim=1),
        torch.flatten(batch_action_label[:, :, :2], start_dim=1),
        dim=-1,
    ))

    results = {
        "uc_action_loss": uc_action_loss,
        "uc_action_waypts_cos_sim": uc_action_waypts_cos_similairity,
        "uc_multi_action_waypts_cos_sim": uc_multi_action_waypts_cos_sim,
        "gc_dist_loss": gc_dist_loss,
        "gc_action_loss": gc_action_loss,
        "gc_action_waypts_cos_sim": gc_action_waypts_cos_similairity,
        "gc_multi_action_waypts_cos_sim": gc_multi_action_waypts_cos_sim,
    }

    return results
    
def sinc_apx(angle):
    return torch.sin(3.141592*angle + 0.000000001)/(3.141592*angle + 0.000000001)
        
def twist_to_pose_diff_torch(v, w, dt):
    """integrate 2D twist to get pose difference.

    Assuming constant velocity during time period `dt`.

    Args:
        v (float): velocity
        w (float): angular velocity
        dt (float): time delta

    """

    theta = -w  * dt
    z = v * dt * sinc_apx(-theta / np.pi)
    x = -v * dt * sinc_apx(-theta / (2 * np.pi)) * torch.sin(-theta / 2)
    return x, z, theta

def robot_pos_model_fix(linear_vel, angular_vel):
    # velocity commands integral
    bs, chorizon = linear_vel.shape
    device = linear_vel.device

    px = []
    pz = []
    pyaw = []
    Tacc = torch.eye(4, 4).unsqueeze(0).repeat(bs,1,1).to(device)
    for i in range(chorizon):
        x, z, yaw = twist_to_pose_diff_torch(linear_vel[:, i], angular_vel[:, i], 0.333)
        Todom = torch.zeros((bs, 4, 4)).to(device)
        Todom[:, 0, 0] = torch.cos(yaw)
        Todom[:, 0, 2] = torch.sin(yaw)
        Todom[:, 1, 1] = 1.0
        Todom[:, 2, 0] = -torch.sin(yaw)
        Todom[:, 2, 2] = torch.cos(yaw)
        Todom[:, 0, 3] = x
        Todom[:, 2, 3] = z
        Todom[:, 3, 3] = 1.0        
        
        Tacc = torch.matmul(Tacc, Todom)
               
        pyaw.append(torch.arctan(Tacc[:, 0, 2]/(Tacc[:, 0, 0] + 0.000000001)))        
        px.append(Tacc[:, 0, 3])
        pz.append(Tacc[:, 2, 3])        
    return px, pz, pyaw    

def train_lelan(
    model: nn.Module,
    ema_model: EMAModel,
    optimizer: Adam,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    project_folder: str,
    epoch: int,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    use_wandb: bool = True,
):
    """
    Train the model for one epoch.

    Args:
        model: model to train
        ema_model: exponential moving average model
        optimizer: optimizer to use
        dataloader: dataloader for training
        transform: transform to use
        device: device to use
        project_folder: folder to save images to
        epoch: current epoch
        print_log_freq: how often to print loss
        wandb_log_freq: how often to log with wandb
        image_log_freq: how often to log images
        num_images_log: number of images to log
        use_wandb: whether to use wandb
    """
    model.train()
    model.eval_text_encoder()
    num_batches = len(dataloader)

    total_loss_logger = Logger("total loss", "train", window_size=print_log_freq)    
    pose_loss_logger = Logger("pose loss", "train", window_size=print_log_freq)
    smooth_loss_logger = Logger("smooth loss", "train", window_size=print_log_freq)    
    loggers = {
        "total loss": total_loss_logger,    
        "pose loss": pose_loss_logger,
        "vel smooth loss": smooth_loss_logger,
    }
    with tqdm.tqdm(dataloader, desc="Train Batch", leave=False) as tepoch:
        for i, data in enumerate(tepoch):
            (
                obs_images, 
                goal_image,
                obj_poses,
                obj_inst,
                goal_pos_norm,                
            ) = data
            
            obs_images_list = torch.split(obs_images, 3, dim=1)
            obs_image = obs_images_list[-1]              
            
            batch_viz_obs_images = TF.resize((255.0*obs_image).type(torch.uint8), VISUALIZATION_IMAGE_SIZE[::-1])
            batch_viz_goal_images = TF.resize((255.0*goal_image).type(torch.uint8), VISUALIZATION_IMAGE_SIZE[::-1])
                            
            batch_obs_images = transform(obs_image).to(device)
            batch_obj_poses = obj_poses.to(device)
            
            batch_obj_inst = clip.tokenize(obj_inst, truncate=True).to(device)          
            
            with torch.no_grad():  
                feat_text = model("text_encoder", inst_ref=batch_obj_inst)
            
            B = batch_obs_images.shape[0]
            
            obsgoal_cond = model("vision_encoder", obs_img=batch_obs_images, feat_text = feat_text.to(dtype=torch.float32))
            linear_vel, angular_vel = model("dist_pred_net", obsgoal_cond=obsgoal_cond)

            px_ref_list, pz_ref_list, ry_ref_list = robot_pos_model_fix(linear_vel, angular_vel)
            px_ref = px_ref_list[-1]
            pz_ref = pz_ref_list[-1]
            ry_ref = ry_ref_list[-1]
 
            last_poses = torch.cat((px_ref.unsqueeze(1), pz_ref.unsqueeze(1)), axis=1)
                                
            dist_loss = nn.functional.mse_loss(last_poses, batch_obj_poses)   
            diff_loss = nn.functional.mse_loss(linear_vel[:,:-1], linear_vel[:,1:]) + nn.functional.mse_loss(angular_vel[:,:-1], angular_vel[:,1:]) 
            
            # Total loss
            loss = dist_loss + 1.0*diff_loss

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update Exponential Moving Average of the model weights
            ema_model.step(model)

            # Logging
            
            loss_cpu = loss.item()
            tepoch.set_postfix(loss=loss_cpu)
            wandb.log({"total loss": loss_cpu})
            wandb.log({"pose loss": dist_loss.item()})
            wandb.log({"vel smooth loss": diff_loss.item()})

            if i % print_log_freq == 0:
                losses = {}
                losses['total loss'] = loss_cpu
                losses['pose loss'] = dist_loss.item()
                losses['vel smooth loss'] = diff_loss.item()                 
                
                for key, value in losses.items():
                    if key in loggers:
                        logger = loggers[key]
                        logger.log_data(value)
            
                data_log = {}
                for key, logger in loggers.items():
                    data_log[logger.full_name()] = logger.latest()
                    if i % print_log_freq == 0 and print_log_freq != 0:
                        print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")

                if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
                    wandb.log(data_log, commit=True)

            if image_log_freq != 0 and i % image_log_freq == 0:
                visualize_lelan_estimation(
                    batch_viz_obs_images,
                    batch_viz_goal_images,
                    obj_poses,
                    obj_inst,
                    linear_vel.cpu(),
                    angular_vel.cpu(),
                    last_poses.cpu(),
                    "train",
                    project_folder,
                    epoch,
                    num_images_log,
                    30,                    
                    use_wandb,
                )

def train_lelan_col(
    model: nn.Module,
    ema_model: EMAModel,
    ema_model_nomad: EMAModel,    
    optimizer: Adam,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    noise_scheduler: DDPMScheduler,
    project_folder: str,
    weight_col_loss: float,    
    epoch: int,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    use_wandb: bool = True,
):
    """
    Train the model for one epoch.

    Args:
        model: model to train
        ema_model: exponential moving average model
        ema_model_nomad: exponential moving average model of pre-trained NoMaD policy for cropped goal image        
        optimizer: optimizer to use
        dataloader: dataloader for training
        transform: transform to use
        device: device to use
        noise_scheduler: noise scheduler to train with 
        project_folder: folder to save images to
        weight_col_loss: weight for collision avoindace loss
        epoch: current epoch
        print_log_freq: how often to print loss
        wandb_log_freq: how often to log with wandb
        image_log_freq: how often to log images
        num_images_log: number of images to log
        use_wandb: whether to use wandb
    """
    #goal_mask_prob = torch.clip(torch.tensor(goal_mask_prob), 0, 1)
    model.train()
    model.eval_text_encoder()
    ema_model_nomad = ema_model_nomad.averaged_model
    ema_model_nomad.eval()    
    num_batches = len(dataloader)

    total_loss_logger = Logger("total loss", "train", window_size=print_log_freq)    
    pose_loss_logger = Logger("pose loss", "train", window_size=print_log_freq)
    smooth_loss_logger = Logger("smooth loss", "train", window_size=print_log_freq)    
    col_loss_logger = Logger("col loss", "train", window_size=print_log_freq)       
    loggers = {
        "total loss": total_loss_logger,    
        "pose loss": pose_loss_logger,
        "vel smooth loss": smooth_loss_logger,
        "col loss": col_loss_logger,        
    }
    with tqdm.tqdm(dataloader, desc="Train Batch", leave=False) as tepoch:
        for i, data in enumerate(tepoch):
            (
                obs_images, 
                goal_image,
                goal_pos,
                obj_inst,
                goal_pos_norm,                
            ) = data
            
            obs_images_list = torch.split(obs_images, 3, dim=1)
            obs_image = obs_images_list[-1]              
            
            batch_viz_obs_images = TF.resize((255.0*obs_image).type(torch.uint8), VISUALIZATION_IMAGE_SIZE[::-1])
            batch_viz_goal_images = TF.resize((255.0*goal_image).type(torch.uint8), VISUALIZATION_IMAGE_SIZE[::-1])
                                                      
            batch_obs_current = transform(obs_image).to(device)

            batch_goal_pos = goal_pos.to(device)
            batch_goal_pos_norm = goal_pos_norm.to(device)      
                        
            batch_obs_images = [transform(TF.resize(obs, (96, 96), antialias=True)) for obs in obs_images_list]
            batch_obs_images = torch.cat(batch_obs_images, dim=1).to(device)
            batch_goal_images = transform(TF.resize(goal_image, (96, 96), antialias=True)).to(device)
            
            batch_obj_inst = clip.tokenize(obj_inst, truncate=True).to(device)          
            
            B = batch_obs_images.shape[0]
            action_mask = torch.ones(B).to(device)
                        
            # split into batches
            batch_obs_images_list = torch.split(batch_obs_images, B, dim=0)
            batch_goal_images_list = torch.split(batch_goal_images, B, dim=0)

            with torch.no_grad():
                select_traj = supervision_from_nomad(
                    ema_model_nomad,
                    noise_scheduler,
                    batch_obs_images,
                    batch_goal_images,
                    batch_viz_obs_images,
                    batch_viz_goal_images,
                    batch_goal_pos_norm,
                    device,
                    project_folder,
                    epoch,
                    B,
                    i,                
                    30,
                    use_wandb,
                    )    
            
            with torch.no_grad():
                feat_text = model("text_encoder", inst_ref=batch_obj_inst)
                                                
            obsgoal_cond = model("vision_encoder", obs_img=batch_obs_images, feat_text = feat_text.to(dtype=torch.float32), current_img=batch_obs_current)
            linear_vel, angular_vel = model("dist_pred_net", obsgoal_cond=obsgoal_cond)

            px_ref_list, pz_ref_list, ry_ref_list = robot_pos_model_fix(linear_vel, angular_vel)
            px_ref = px_ref_list[-1]
            pz_ref = pz_ref_list[-1]
            ry_ref = ry_ref_list[-1]
            last_poses = torch.cat((px_ref.unsqueeze(1), pz_ref.unsqueeze(1)), axis=1)

            #transformation from camera coordinate to robot coordinate
            px_ref_listx = []
            pz_ref_listx = []
            for it in range(8):
                px_ref_listx.append(px_ref_list[it].unsqueeze(1).unsqueeze(2))
                pz_ref_listx.append(pz_ref_list[it].unsqueeze(1).unsqueeze(2))
            traj_policy = torch.concat((torch.concat(pz_ref_listx, axis=1), -torch.concat(px_ref_listx, axis=1)), axis=2)
                                
            dist_loss = nn.functional.mse_loss(last_poses, batch_goal_pos)   
            diff_loss = nn.functional.mse_loss(linear_vel[:,:-1], linear_vel[:,1:]) + nn.functional.mse_loss(angular_vel[:,:-1], angular_vel[:,1:]) 
            
            mask_nomad = (batch_goal_pos[:,1:2] > 1.0).float().unsqueeze(1).repeat(1,8,2)
            mask_dist = (~(batch_goal_pos[:,1:2] > 1.0)).float()
            sum_dist = mask_dist.sum()            
            col_loss = nn.functional.mse_loss(mask_nomad*traj_policy, 0.12*mask_nomad*select_traj)*float(B)/(float(B) - sum_dist.float() + 1e-7) #0.12 is de-normalization
            
            loss = 1.0*dist_loss + 1.0*diff_loss + weight_col_loss*col_loss

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update Exponential Moving Average of the model weights
            ema_model.step(model)

            # Logging
            
            loss_cpu = loss.item()
            tepoch.set_postfix(loss=loss_cpu)
            wandb.log({"total loss": loss_cpu})
            wandb.log({"pose loss": dist_loss.item()})
            wandb.log({"vel smooth loss": diff_loss.item()})
            wandb.log({"col loss": col_loss.item()})
            
            if i % print_log_freq == 0:
                losses = {}
                losses['total loss'] = loss_cpu
                losses['pose loss'] = dist_loss.item()
                losses['vel smooth loss'] = diff_loss.item()                 
                losses['col loss'] = col_loss.item()       
                                
                for key, value in losses.items():
                    if key in loggers:
                        logger = loggers[key]
                        logger.log_data(value)
            
                data_log = {}
                for key, logger in loggers.items():
                    data_log[logger.full_name()] = logger.latest()
                    if i % print_log_freq == 0 and print_log_freq != 0:
                        print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")

                if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
                    wandb.log(data_log, commit=True)

            if image_log_freq != 0 and i % image_log_freq == 0:
                visualize_lelan_col_estimation(
                    batch_viz_obs_images,
                    batch_viz_goal_images,
                    goal_pos,
                    obj_inst,
                    linear_vel.cpu(),
                    angular_vel.cpu(),
                    last_poses.cpu(),
                    (0.12*select_traj).cpu(),
                    "train",
                    project_folder,
                    epoch,
                    num_images_log,
                    30,                    
                    use_wandb,
                )

def train_nomad(
    model: nn.Module,
    ema_model: EMAModel,
    optimizer: Adam,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    noise_scheduler: DDPMScheduler,
    goal_mask_prob: float,
    project_folder: str,
    epoch: int,
    alpha: float = 1e-4,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    use_wandb: bool = True,
):
    """
    Train the model for one epoch.

    Args:
        model: model to train
        ema_model: exponential moving average model
        optimizer: optimizer to use
        dataloader: dataloader for training
        transform: transform to use
        device: device to use
        noise_scheduler: noise scheduler to train with 
        project_folder: folder to save images to
        epoch: current epoch
        alpha: weight of action loss
        print_log_freq: how often to print loss
        image_log_freq: how often to log images
        num_images_log: number of images to log
        use_wandb: whether to use wandb
    """
    goal_mask_prob = torch.clip(torch.tensor(goal_mask_prob), 0, 1)
    model.train()
    num_batches = len(dataloader)

    uc_action_loss_logger = Logger("uc_action_loss", "train", window_size=print_log_freq)
    uc_action_waypts_cos_sim_logger = Logger(
        "uc_action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    uc_multi_action_waypts_cos_sim_logger = Logger(
        "uc_multi_action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    gc_dist_loss_logger = Logger("gc_dist_loss", "train", window_size=print_log_freq)
    gc_action_loss_logger = Logger("gc_action_loss", "train", window_size=print_log_freq)
    gc_action_waypts_cos_sim_logger = Logger(
        "gc_action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    gc_multi_action_waypts_cos_sim_logger = Logger(
        "gc_multi_action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    loggers = {
        "uc_action_loss": uc_action_loss_logger,
        "uc_action_waypts_cos_sim": uc_action_waypts_cos_sim_logger,
        "uc_multi_action_waypts_cos_sim": uc_multi_action_waypts_cos_sim_logger,
        "gc_dist_loss": gc_dist_loss_logger,
        "gc_action_loss": gc_action_loss_logger,
        "gc_action_waypts_cos_sim": gc_action_waypts_cos_sim_logger,
        "gc_multi_action_waypts_cos_sim": gc_multi_action_waypts_cos_sim_logger,
    }
    with tqdm.tqdm(dataloader, desc="Train Batch", leave=False) as tepoch:
        for i, data in enumerate(tepoch):
            (
                obs_image, 
                goal_image,
                actions,
                distance,
                goal_pos,
                dataset_idx,
                action_mask, 
            ) = data

            
            obs_images = torch.split(obs_image, 3, dim=1)
            batch_viz_obs_images = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE[::-1])
            batch_viz_goal_images = TF.resize(goal_image, VISUALIZATION_IMAGE_SIZE[::-1])
            batch_obs_images = [transform(obs) for obs in obs_images]
            batch_obs_images = torch.cat(batch_obs_images, dim=1).to(device)
            batch_goal_images = transform(goal_image).to(device)
            action_mask = action_mask.to(device)

            B = actions.shape[0]

            # Generate random goal mask
            goal_mask = (torch.rand((B,)) < goal_mask_prob).long().to(device)
            obsgoal_cond = model("vision_encoder", obs_img=batch_obs_images, goal_img=batch_goal_images, input_goal_mask=goal_mask)
            
            # Get distance label
            distance = distance.float().to(device)

            deltas = get_delta(actions)         
            ndeltas = normalize_data(deltas, ACTION_STATS)         
            naction = from_numpy(ndeltas).to(device)                 
            assert naction.shape[-1] == 2, "action dim must be 2"

            # Predict distance
            dist_pred = model("dist_pred_net", obsgoal_cond=obsgoal_cond)
            dist_loss = nn.functional.mse_loss(dist_pred.squeeze(-1), distance)
            dist_loss = (dist_loss * (1 - goal_mask.float())).mean() / (1e-2 +(1 - goal_mask.float()).mean())

            # Sample noise to add to actions
            noise = torch.randn(naction.shape, device=device)

            # Sample a diffusion iteration for each data point
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (B,), device=device
            ).long()

            # Add noise to the clean images according to the noise magnitude at each diffusion iteration
            noisy_action = noise_scheduler.add_noise(
                naction, noise, timesteps)         
                        
            # Predict the noise residual
            noise_pred = model("noise_pred_net", sample=noisy_action, timestep=timesteps, global_cond=obsgoal_cond)

            def action_reduce(unreduced_loss: torch.Tensor):
                # Reduce over non-batch dimensions to get loss per batch element
                while unreduced_loss.dim() > 1:
                    unreduced_loss = unreduced_loss.mean(dim=-1)
                assert unreduced_loss.shape == action_mask.shape, f"{unreduced_loss.shape} != {action_mask.shape}"
                return (unreduced_loss * action_mask).mean() / (action_mask.mean() + 1e-2)

            # L2 loss
            diffusion_loss = action_reduce(F.mse_loss(noise_pred, noise, reduction="none"))
            
            # Total loss
            loss = alpha * dist_loss + (1-alpha) * diffusion_loss # mse between ground truth noise and predicted noise

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update Exponential Moving Average of the model weights
            ema_model.step(model)

            # Logging
            
            loss_cpu = loss.item()
            tepoch.set_postfix(loss=loss_cpu)
            wandb.log({"total_loss": loss_cpu})
            wandb.log({"dist_loss": dist_loss.item()})
            wandb.log({"diffusion_loss": diffusion_loss.item()})


            if i % print_log_freq == 0:
                losses = _compute_losses_nomad(
                            ema_model.averaged_model,
                            noise_scheduler,
                            batch_obs_images,
                            batch_goal_images,
                            distance.to(device),
                            actions.to(device),
                            device,
                            action_mask.to(device),
                        )
                
                for key, value in losses.items():
                    if key in loggers:
                        logger = loggers[key]
                        logger.log_data(value.item())
            
                data_log = {}
                for key, logger in loggers.items():
                    data_log[logger.full_name()] = logger.latest()
                    if i % print_log_freq == 0 and print_log_freq != 0:
                        print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")

                if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
                    wandb.log(data_log, commit=True)

            if image_log_freq != 0 and i % image_log_freq == 0:
                visualize_diffusion_action_distribution(
                    ema_model.averaged_model,
                    noise_scheduler,
                    batch_obs_images,
                    batch_goal_images,
                    batch_viz_obs_images,
                    batch_viz_goal_images,
                    actions,
                    distance,
                    goal_pos,
                    device,
                    "train",
                    project_folder,
                    epoch,
                    num_images_log,
                    30,
                    use_wandb,
                )

###
def train_exaug_dist_gnm_delay(
    model: nn.Module,
    ema_model: EMAModel,
    optimizer: Adam,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    latest_path: str,
    dataloader: DataLoader,
    dataloader_sub: DataLoader,
    transform: transforms,
    device: torch.device,
    project_folder: str,
    epoch: int,
    sacson: bool,
    no_emamodel: bool,
    model_depth,
    model_pedtraj,
    device2,      
    len_traj_pred: int,       
    alpha: float = 1e-4,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    use_wandb: bool = True,   
):
    """
    Train the model for one epoch.

    Args:
        model: model to train
        ema_model: exponential moving average model
        optimizer: optimizer to use
        dataloader: dataloader for training
        transform: transform to use
        device: device to use
        noise_scheduler: noise scheduler to train with 
        project_folder: folder to save images to
        epoch: current epoch
        alpha: weight of action loss
        print_log_freq: how often to print loss
        image_log_freq: how often to log images
        num_images_log: number of images to log
        use_wandb: whether to use wandb
    """
    #goal_mask_prob = torch.clip(torch.tensor(goal_mask_prob), 0, 1)
    model.train()
    num_batches = len(dataloader)

    total_loss_logger = Logger("total_loss", "train", window_size=print_log_freq)
    dist_loss_logger = Logger("dist_loss", "train", window_size=print_log_freq)
    distall_loss_logger = Logger("distall_loss", "train", window_size=print_log_freq)    
    smooth_loss_logger = Logger("smooth_loss", "train", window_size=print_log_freq)
    geo_loss_logger = Logger("geo_loss", "train", window_size=print_log_freq)
    social_loss_logger = Logger("social_loss", "train", window_size=print_log_freq)
    personal_loss_logger = Logger("personal_loss", "train", window_size=print_log_freq)
    disttemp_loss_logger = Logger("disttemp_loss", "train", window_size=print_log_freq)
        
    loggers = {
        "total_loss": total_loss_logger,
        "dist_loss": dist_loss_logger,
        "distall_loss": distall_loss_logger,        
        "smooth_loss": smooth_loss_logger,
        "geo_loss": geo_loss_logger,
        "social_loss": social_loss_logger,
        "personal_loss": personal_loss_logger,        
        "disttemp_loss": disttemp_loss_logger,          
    }
    
    mask_360 = np.loadtxt(open("./mask_360view.csv", "rb"), delimiter=",", skiprows=0)           
    mask_360_resize = np.repeat(np.expand_dims(cv2.resize(mask_360, (832, 128)), 0), 3, 0).astype(np.float32)
    mask_360_torch = torch.from_numpy(mask_360_resize[:,:,0:416]).unsqueeze(0).to(device2)
    dataloader_sub_iter = iter(dataloader_sub)

    linear_vel_old = 0.5*torch.rand(100, 8).float().to(device)
    angular_vel_old = 1.0*torch.rand(100, 8).float().to(device)
                  
    with tqdm.tqdm(dataloader, desc="Train Batch", leave=False) as tepoch:
        for i, data in enumerate(tepoch):
            (
                obs_image, 
                goal_image,
                current_image,                
                actions,
                distance,
                goal_pos,
                local_goal_mat,
                local_yaw,              
                _,
                _,
                id_num,
                action_mask,
                ped_list,
                ped_list_raw,
                ped_list_no_trans,
                robot_list,
            ) = data
            try:
                (
                    obs_image_sub,
                    goal_image_sub,
                    action_label_sub,
                    dist_label_sub,
                    goal_pos_sub,
                    dataset_index_sub,
                    action_mask_sub,
                    current_image_depth_sub,
                    geoloss_range_sub,
                    local_goal_mat_sub,
                    local_yaw_sub,
                ) = next(dataloader_sub_iter)                
            except StopIteration:
                dataloader_sub_iter = iter(dataloader_sub) 
                (
                    obs_image_sub,
                    goal_image_sub,
                    action_label_sub,
                    dist_label_sub,
                    goal_pos_sub,
                    dataset_index_sub,
                    action_mask_sub,
                    current_image_depth_sub,
                    geoloss_range_sub,
                    local_goal_mat_sub,
                    local_yaw_sub,
                ) = next(dataloader_sub_iter)   
                
            Bf, _, _, _ = goal_image.size()
            Bg, _, _, _ = goal_image_sub.size()

            obs_images_sub = torch.split(obs_image_sub, 3, dim=1)
            viz_obs_image_sub = TF.resize(obs_images_sub[-1], VISUALIZATION_IMAGE_SIZE)
            viz_obs_image_past_sub = TF.resize(obs_images_sub[0], VISUALIZATION_IMAGE_SIZE[::-1])
            obs_images_sub = [transform(obs_image_sub).to(device) for obs_image_sub in obs_images_sub]
            obs_image_sub = torch.cat(obs_images_sub, dim=1)
            
            viz_goal_image_sub = TF.resize(goal_image_sub, VISUALIZATION_IMAGE_SIZE[::-1])
            current_image_depth_sub = current_image_depth_sub.to(device2)
            goal_image_sub = transform(goal_image_sub).to(device)

            dist_label_sub = dist_label_sub.to(device)
            action_label_sub = action_label_sub.to(device)
            action_mask_sub = action_mask_sub.to(device)
            local_goal_mat_sub = local_goal_mat_sub.to(device)
            local_yaw_sub = local_yaw_sub.to(device)

            ang_yaw_sub = []
            for iy in range(Bg):
                if local_yaw_sub[iy] % (2*3.14) > 3.14:
                    ang_yaw_sub.append(local_yaw_sub[iy] % (2*3.14) - 2.0*3.14)
                else:
                    ang_yaw_sub.append(local_yaw_sub[iy] % (2*3.14))            
            ang_yaw_sub_tensor = torch.tensor(ang_yaw_sub).to(device)
                                    
            distance_sub = dist_label_sub.float().to(device)
            fargoal_mask_sub = ((torch.abs(local_goal_mat_sub[:, 0,2]) < 2.0) * (torch.abs(local_goal_mat_sub[:, 1,2]) < 2.0)).to(device) * (torch.abs(ang_yaw_sub_tensor) < 2.0)                        
            goal_mask_sub = (dist_label_sub > 0.1) * fargoal_mask_sub
                                    
            local_goal_mat_sub[:, 0,2] *= 2.0                    
            local_goal_mat_sub[:, 1,2] *= 2.0  
            local_goal_vec_sub = local_goal_mat_sub.unsqueeze(1).repeat(1,8,1,1) 
                                                
            current_image_depth = (current_image.to(device2))*mask_360_torch
            current_image_depth = current_image_depth.to(device2)
            
            obs_images = torch.split(obs_image, 3, dim=1) 
            viz_obs_image = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE[::-1])
            viz_obs_image_past = TF.resize(obs_images[0], VISUALIZATION_IMAGE_SIZE[::-1])            
            obs_images = [transform(obs_image).to(device) for obs_image in obs_images]
            obs_image = torch.cat(obs_images, dim=1)
                        
            combined_current_image_depth = torch.cat((current_image_depth, current_image_depth_sub), axis=0)        
            with torch.no_grad():
                #depth estimation
                proj_3d, outputs = model_depth.forward(combined_current_image_depth) #for depth360   

            batch_3d_point_cpu = proj_3d.cpu()
            batch_3d_point = batch_3d_point_cpu.to(device)   
            

            ang_yaw = []
            for iy in range(Bf):
                if local_yaw[iy] % (2*3.14) > 3.14:
                    ang_yaw.append(local_yaw[iy] % (2*3.14) - 2.0*3.14)
                else:
                    ang_yaw.append(local_yaw[iy] % (2*3.14))
            
            ang_yaw_tensor = torch.tensor(ang_yaw).to(device)

            # Get distance label
            distance_metric = torch.sqrt(goal_pos.to(device)[:,0]**2 + goal_pos.to(device)[:,1]**2)
            fargoal_mask = ((torch.abs(local_goal_mat[:, 0,2]) < 5.0) * (torch.abs(local_goal_mat[:, 1,2]) < 5.0)).to(device)
            
            distance = distance.float().to(device)
            goal_mask = (distance > 0.1) * fargoal_mask
            goal_mask_zero = distance > 0.1

            for ig in range(Bf):
                if not goal_mask_zero[ig]:
                    distance[ig] = 20
                    igr = random.randint(0, Bf-1) 
                    while ig == igr:
                        igr = random.randint(0, Bf-1) 
                    goal_image[ig] = goal_image[igr]
                    #print(ig, igr)

            viz_goal_image = TF.resize(goal_image, VISUALIZATION_IMAGE_SIZE[::-1])
            batch_goal_pos = goal_pos.to(device)
            goal_image = transform(goal_image).to(device)

            combined_obs_image = torch.cat((obs_image, obs_image_sub), axis=0)
            combined_goal_image = torch.cat((goal_image, goal_image_sub), axis=0)

            rsize = torch.rand(Bf+Bg, 1, 1).to(device) #robot radius : 0 -- 1.0 m
            delay = torch.randint(0, 5, (Bf+Bg, 1, 1)).to(device)     

            cs = random.randint(0,2)
            linear_vel_old_p = linear_vel_old[:, cs:cs+6]
            angular_vel_old_p = angular_vel_old[:, cs:cs+6]

            vel_past = torch.cat((linear_vel_old_p, angular_vel_old_p), axis=1).unsqueeze(2)                                  
            linear_vel, angular_vel, dist_temp = model(combined_obs_image, combined_goal_image, rsize, delay, vel_past)                           

            for ig in range(Bf+Bg):
                linear_vel_old_p[ig, delay[ig,0,0]:6] *= 0.0
                angular_vel_old_p[ig, delay[ig,0,0]:6] *= 0.0
                                
            linear_vel_d = torch.cat((linear_vel_old_p, linear_vel), axis=1)
            angular_vel_d = torch.cat((angular_vel_old_p, angular_vel), axis=1)            
            linear_vel_old = linear_vel.detach()
            angular_vel_old = angular_vel.detach()
               
            px_ref_list, pz_ref_list, ry_ref_list = robot_pos_model_fix(linear_vel_d, angular_vel_d)
            px_ref = px_ref_list[-1]
            pz_ref = pz_ref_list[-1]
            ry_ref = ry_ref_list[-1]
            last_poses = torch.cat((pz_ref.unsqueeze(1), -px_ref.unsqueeze(1)), axis=1) #from camera coordinate to robot local coordinate
            
            mat_1 = torch.cat((torch.cos(-ry_ref).unsqueeze(1), -torch.sin(-ry_ref).unsqueeze(1), 2.0 * pz_ref.unsqueeze(1)), axis=1)
            mat_2 = torch.cat((torch.sin(-ry_ref).unsqueeze(1), torch.cos(-ry_ref).unsqueeze(1), -2.0 * px_ref.unsqueeze(1)), axis=1)
            mat_3 = torch.cat((torch.zeros(Bf+Bg,1), torch.zeros(Bf+Bg,1), torch.ones(Bf+Bg,1)), axis=1).to(device)   
            last_pose_mat = torch.cat((mat_1.unsqueeze(1), mat_2.unsqueeze(1), mat_3.unsqueeze(1)), axis=1)
            
            robot_traj_list = []
            for ip in range(len(px_ref_list)):
                mat_1 = torch.cat((torch.cos(-ry_ref_list[ip]).unsqueeze(1), -torch.sin(-ry_ref_list[ip]).unsqueeze(1), 2.0 * pz_ref_list[ip].unsqueeze(1)), axis=1)
                mat_2 = torch.cat((torch.sin(-ry_ref_list[ip]).unsqueeze(1), torch.cos(-ry_ref_list[ip]).unsqueeze(1), -2.0 * px_ref_list[ip].unsqueeze(1)), axis=1)
                mat_3 = torch.cat((torch.zeros(Bf+Bg,1), torch.zeros(Bf+Bg,1), torch.ones(Bf+Bg,1)), axis=1).to(device)   
                mat_combine = torch.cat((mat_1.unsqueeze(1), mat_2.unsqueeze(1), mat_3.unsqueeze(1)), axis=1)
                robot_traj_list.append(mat_combine.unsqueeze(1))
            robot_traj_vec = torch.cat(robot_traj_list, axis=1)
                                 
            local_goal_mat[:, 0,2] *= 2.0                    
            local_goal_mat[:, 1,2] *= 2.0  
            local_goal_vec = local_goal_mat.unsqueeze(1).repeat(1,8,1,1) 
                       
            combined_local_goal_mat = torch.cat((local_goal_mat.to(device), local_goal_mat_sub), axis=0)
            combined_local_goal_vec = torch.cat((local_goal_vec.to(device), local_goal_vec_sub), axis=0)
            combined_distance = torch.cat((distance, distance_sub), axis=0)
            combined_goal_mask = torch.cat((goal_mask, goal_mask_sub), axis=0)
            
            geoloss_range = torch.cat((0.0*torch.ones(Bf,1), -0.3*torch.ones(Bf,1)), axis=1)
            combined_geoloss_range = torch.cat((geoloss_range, geoloss_range_sub), axis=0).to(device)
             
            dist_loss = nn.functional.mse_loss(last_pose_mat[combined_goal_mask], combined_local_goal_mat.to(device)[combined_goal_mask])      
            dist_loss_f = nn.functional.mse_loss(last_pose_mat[0:50][combined_goal_mask[0:50]], combined_local_goal_mat.to(device)[0:50][combined_goal_mask[0:50]])    
            dist_loss_g = nn.functional.mse_loss(last_pose_mat[50:100][combined_goal_mask[50:100]], combined_local_goal_mat.to(device)[50:100][combined_goal_mask[50:100]])    
       
            distall_loss = nn.functional.mse_loss(robot_traj_vec[combined_goal_mask, 6:14], combined_local_goal_vec.to(device)[combined_goal_mask])                                    
            diff_loss = nn.functional.mse_loss(linear_vel[:,:-1][combined_goal_mask], linear_vel[:,1:][combined_goal_mask]) + nn.functional.mse_loss(angular_vel[:,:-1][combined_goal_mask], angular_vel[:,1:][combined_goal_mask]) 

            # Predict distance          
            dist_temp_loss = F.mse_loss(dist_temp.squeeze(-1), combined_distance)
            
            norm = 10.0
            
            if sacson:
                ped_past = torch.cat((-torch.flip(ped_list[:,:,1], dims=[1]), torch.flip(ped_list[:,:,0], dims=[1])), axis=1).to(device)  
                robot_past = torch.cat((-torch.flip(robot_list[:,:,1], dims=[1]), torch.flip(robot_list[:,:,0], dims=[1])), axis=1).to(device)
                
                robot_future_x = []
                robot_future_z = []
                for ir in range(len(px_ref_list)):
                    robot_future_x.append(px_ref_list[ir].unsqueeze(1))
                    robot_future_z.append(pz_ref_list[ir].unsqueeze(1))                    
                robot_future = torch.cat((torch.cat(robot_future_x, axis=1), torch.cat(robot_future_z, axis=1)), axis=1).to(device)  
                              
                ped_past_c = torch.clamp(ped_past, min=-10.0, max=10.0)/norm
                robot_past_c = torch.clamp(robot_past, min=-10.0, max=10.0)/norm
                robot_future_c = torch.clamp(robot_future, min=-10.0, max=10.0)/norm          
                flag_ped = (ped_past_c[:,0] != 0.0) & (ped_past_c[:,8] != 0.0)
                
                robot_past_c = 0.333*robot_past_c #frodbot is much faster than Vizbot.
                delta_est_ped_traj = model_pedtraj(ped_past_c, robot_past_c, robot_future_c)
                delta_est_ped_traj_zero = model_pedtraj(ped_past_c, robot_past_c, 0.0*robot_future_c)

                traj_x = torch.cumsum(delta_est_ped_traj[:,0:8]/norm, dim=1) + ped_past_c[:,0:1].repeat(1,8)
                traj_y = torch.cumsum(delta_est_ped_traj[:,8:16]/norm, dim=1) + ped_past_c[:,8:9].repeat(1,8)           
                est_ped_traj = torch.clamp(torch.cat((traj_x, traj_y), axis=1), min=-10.0, max=10.0)

                delta_est_ped_traj_zero_detach = delta_est_ped_traj_zero.detach()
                traj_xz = torch.cumsum(delta_est_ped_traj_zero_detach[:,0:8]/norm, dim=1) + ped_past_c[:,0:1].repeat(1,8)
                traj_yz = torch.cumsum(delta_est_ped_traj_zero_detach[:,8:16]/norm, dim=1) + ped_past_c[:,8:9].repeat(1,8)           
                est_ped_traj_zeros = torch.clamp(torch.cat((traj_xz, traj_yz), axis=1), min=-10.0, max=10.0)
                
                social_loss = nn.functional.mse_loss(est_ped_traj[flag_ped*combined_goal_mask], est_ped_traj_zeros[flag_ped*combined_goal_mask])
                
                max_pl = rsize.squeeze(1).repeat(1,8) + 0.5
                min_pl = rsize.squeeze(1).repeat(1,8) * 0.0
                personal_loss = nn.functional.mse_loss(max_pl[flag_ped*combined_goal_mask], torch.clamp(torch.sqrt((norm*robot_future_c[flag_ped*combined_goal_mask][:,0:8] - norm*est_ped_traj[flag_ped*combined_goal_mask][:,0:8])**2 + (norm*robot_future_c[flag_ped*combined_goal_mask][:,8:16] - norm*est_ped_traj[flag_ped*combined_goal_mask][:,8:16])**2), min=min_pl[flag_ped*combined_goal_mask], max= max_pl[flag_ped*combined_goal_mask]))
            else:
                est_ped_traj = torch.zeros(Bf+Bg, 16)
                est_ped_traj_zeros = torch.zeros(Bf+Bg, 16)      
                ped_past_c = torch.zeros(Bf+Bg, 16)
                robot_past_c = torch.zeros(Bf+Bg, 16)
                social_loss = nn.functional.mse_loss(est_ped_traj, est_ped_traj_zeros)
                personal_loss = nn.functional.mse_loss(est_ped_traj, est_ped_traj_zeros)

            PC3D = []
            for j in range(len_traj_pred):
                px_ref = px_ref_list[6+j]
                pz_ref = pz_ref_list[6+j]
                ry_ref = ry_ref_list[6+j]                

                Tod = torch.zeros((Bf+Bg, 4, 4)).to(device)
                Tod[:, 0, 0] = torch.cos(ry_ref)
                Tod[:, 0, 2] = torch.sin(ry_ref)
                Tod[:, 1, 1] = 1.0
                Tod[:, 2, 0] = -torch.sin(ry_ref)
                Tod[:, 2, 2] = torch.cos(ry_ref)
                Tod[:, 0, 3] = px_ref
                Tod[:, 2, 3] = pz_ref
                Tod[:, 3, 3] = 1.0

                Ttrans = torch.inverse(Tod)[:, :3, :]               
                batch_3d_point_x = torch.cat((batch_3d_point.view(Bf+Bg, 3, -1), torch.ones(Bf+Bg,1,416*128).to(device)), axis=1)
                cam_points_trans = torch.matmul(Ttrans, batch_3d_point_x).view(Bf+Bg, 3, 128, 416)
                PC3D.append(cam_points_trans.unsqueeze(1))                                                  
            
            PC3D_cat = torch.cat(PC3D, axis=1)                    
            loss_geo = geometry_criterion_range(PC3D_cat[combined_goal_mask], rsize[:,:,0][combined_goal_mask], len_traj_pred, combined_geoloss_range[combined_goal_mask], device)
            loss = 4.0*dist_loss + 0.4*distall_loss + 0.5*diff_loss + 10.0*loss_geo + 100.0*social_loss + 10.0*personal_loss + 0.001*dist_temp_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Logging            
            loss_cpu = loss.item()
            tepoch.set_postfix(loss=loss_cpu)
            wandb.log({"total_loss": loss_cpu})
            wandb.log({"dist_loss": dist_loss.item()})
            wandb.log({"distall_loss": distall_loss.item()})            
            wandb.log({"smooth_loss": diff_loss.item()})
            wandb.log({"geo_loss": loss_geo.item()})
            wandb.log({"social_loss": social_loss.item()})
            wandb.log({"personal_loss": personal_loss.item()})
            wandb.log({"disttemp_loss": dist_temp_loss.item()}) 
            
            if epoch == 0 and i == 2000:
                lr_scheduler.step()
                current_lrs = get_current_lr(optimizer)  
                print(i, current_lrs) 
            if i % 10000 == 0 and i != 0:
                lr_scheduler.step()
                current_lrs = get_current_lr(optimizer)  
                print(i, current_lrs) 
                
            if i % 500 == 0 and i != 0:
                if no_emamodel:
                    numbered_path = os.path.join(project_folder, f"ema_{epoch}.pth")
                    torch.save(ema_model.averaged_model.state_dict(), numbered_path)

                numbered_path = os.path.join(project_folder, f"{epoch}.pth")
                torch.save(model.state_dict(), numbered_path)
                torch.save(model.state_dict(), latest_path)

                # save optimizer
                numbered_path = os.path.join(project_folder, f"optimizer_{epoch}.pth")
                latest_optimizer_path = os.path.join(project_folder, f"optimizer_latest.pth")
                torch.save(optimizer.state_dict(), latest_optimizer_path)

                # save scheduler
                numbered_path = os.path.join(project_folder, f"scheduler_{epoch}.pth")
                latest_scheduler_path = os.path.join(project_folder, f"scheduler_latest.pth")
                torch.save(lr_scheduler.state_dict(), latest_scheduler_path)
        
            #if False:
            if i % print_log_freq == 0:
                losses = {}
                losses['total_loss'] = loss_cpu
                losses['dist_loss'] = dist_loss.item()
                losses['distall_loss'] = distall_loss.item()                
                losses['smooth_loss'] = diff_loss.item()                 
                losses['geo_loss'] = loss_geo.item()
                losses['social_loss'] = social_loss.item()                 
                losses['personal_loss'] = personal_loss.item()  
                losses['disttemp_loss'] = dist_temp_loss.item()  
                                                                
                for key, value in losses.items():
                    if key in loggers:
                        logger = loggers[key]
                        logger.log_data(value)
            
                data_log = {}
                for key, logger in loggers.items():
                    data_log[logger.full_name()] = logger.latest()
                    if i % print_log_freq == 0 and print_log_freq != 0:
                        print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")

                if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
                    wandb.log(data_log, commit=True)
            
            if image_log_freq != 0 and i % image_log_freq == 0:
                visualize_exaug_delay_estimation(
                    viz_obs_image, 
                    viz_obs_image_past,                     
                    viz_goal_image,
                    batch_3d_point,
                    goal_pos,
                    local_yaw,
                    linear_vel_d.cpu(),
                    angular_vel_d.cpu(),
                    norm*ped_past_c.cpu(),
                    norm*est_ped_traj.cpu(),
                    norm*est_ped_traj_zeros.cpu(),
                    norm*robot_past_c.cpu(),
                    last_poses.cpu(),
                    rsize.cpu(),
                    "train",
                    project_folder,
                    epoch,
                    num_images_log,
                    30,                    
                    use_wandb,
                    )

###
def train_il_dist_gnm(
    model: nn.Module,
    ema_model: EMAModel,
    optimizer: Adam,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    latest_path: str,
    dataloader: DataLoader,
    dataloader_sub: DataLoader,    
    transform: transforms,
    device: torch.device,
    project_folder: str,
    epoch: int,
    sacson: bool,
    no_emamodel: bool,
    #model_depth,
    #model_pedtraj,
    #device2,      
    len_traj_pred: int,       
    alpha: float = 1e-4,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    use_wandb: bool = True,   
):
    """
    Train the model for one epoch.

    Args:
        model: model to train
        ema_model: exponential moving average model
        optimizer: optimizer to use
        dataloader: dataloader for training
        transform: transform to use
        device: device to use
        noise_scheduler: noise scheduler to train with 
        project_folder: folder to save images to
        epoch: current epoch
        alpha: weight of action loss
        print_log_freq: how often to print loss
        image_log_freq: how often to log images
        num_images_log: number of images to log
        use_wandb: whether to use wandb
    """
    model.train()
    num_batches = len(dataloader)

    total_loss_logger = Logger("total_loss", "train", window_size=print_log_freq)
    dist_loss_logger = Logger("dist_loss", "train", window_size=print_log_freq)
    action_loss_logger = Logger("action_loss", "train", window_size=print_log_freq)    

    loggers = {
        "total_loss": total_loss_logger,
        "dist_loss": dist_loss_logger,
        "action_loss": action_loss_logger,                
    }       
    dataloader_sub_iter = iter(dataloader_sub)           
       
    with tqdm.tqdm(dataloader, desc="Train Batch", leave=False) as tepoch:
        for i, data in enumerate(tepoch):
            
            (
                obs_image, 
                goal_image,
                goal_image2,                
                current_image,                
                actions,
                distance,
                goal_pos,
                local_goal_mat,
                local_yaw,              
                which_dataset,
                obs_image_future,
                id_num,
                action_mask,
                ped_list,
                ped_list_raw,
                ped_list_no_trans,
                robot_list,
            ) = data         
            try:
                (
                    obs_image_sub,
                    goal_image_sub,
                    action_label_sub,
                    dist_label_sub,
                    goal_pos_sub,
                    dataset_index_sub,
                    action_mask_sub,
                ) = next(dataloader_sub_iter)
            except StopIteration:
                dataloader_sub_iter = iter(dataloader_sub) 
                (
                    obs_image_sub,
                    goal_image_sub,
                    action_label_sub,
                    dist_label_sub,
                    goal_pos_sub,
                    dataset_index_sub,
                    action_mask_sub,
                ) = next(dataloader_sub_iter)        
    
            Bsub, _, H, W = obs_image_sub.size()  

            obs_images_sub = torch.split(obs_image_sub, 3, dim=1)
            viz_obs_image_sub = TF.resize(obs_images_sub[-1], VISUALIZATION_IMAGE_SIZE)
            viz_obs_image_past_sub = TF.resize(obs_images_sub[0], VISUALIZATION_IMAGE_SIZE[::-1])
            obs_images_sub = [transform(obs_image_sub).to(device) for obs_image_sub in obs_images_sub]
            obs_image_sub = torch.cat(obs_images_sub, dim=1)

            viz_goal_image_sub = TF.resize(goal_image_sub, VISUALIZATION_IMAGE_SIZE[::-1])
        
            goal_image_sub = transform(goal_image_sub).to(device)
            dist_label_sub = dist_label_sub.to(device)
            action_label_sub = action_label_sub.to(device)
            action_mask_sub = action_mask_sub.to(device)
            goal_mask_sub = dist_label_sub > -1.0
                    
            if psutil.virtual_memory().percent > 90.0:
                print("RAM usage (%)", psutil.virtual_memory().percent)
                break            
            B, _, H, W = obs_image.size()  
            
            obs_images = torch.split(obs_image, 3, dim=1)
            viz_obs_image = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE)
            viz_obs_image_past = TF.resize(obs_images[0], VISUALIZATION_IMAGE_SIZE[::-1])     
            
            obs_images_future = torch.split(obs_image_future, 3, dim=1)                  
            obs_images = [transform(obs_image).to(device) for obs_image in obs_images]
            obs_image = torch.cat(obs_images, dim=1)
            actions = actions.to(device)
            action_mask = action_mask.to(device)
            batch_goal_pos = goal_pos.to(device)
        
            # Get distance label
            distance = distance.float().to(device)
            goal_mask = distance > 0.1                        

            for ig in range(B):
                if not goal_mask[ig]:
                    distance[ig] = 20
                    igr = random.randint(0, B-1) 
                    while ig == igr:
                        igr = random.randint(0, B-1) 
                    goal_image[ig] = goal_image[igr]

            viz_goal_image = TF.resize(goal_image, VISUALIZATION_IMAGE_SIZE[::-1])
            goal_image = transform(goal_image).to(device)         
            
            combined_obs_image = torch.cat((obs_image, obs_image_sub), axis=0)
            combined_goal_image = torch.cat((goal_image, goal_image_sub), axis=0)            
            combined_actions = torch.cat((actions, action_label_sub), axis=0)   
             
            combined_distance = torch.cat((distance, dist_label_sub), axis=0)   
            combined_goal_mask = torch.cat((goal_mask, goal_mask_sub), axis=0)                   
            combined_action_mask = torch.cat((action_mask, action_mask_sub), axis=0) 

            combined_viz_obs_image = torch.cat((viz_obs_image, viz_obs_image_sub), axis=0)                   
            combined_viz_obs_image_past = torch.cat((viz_obs_image_past, viz_obs_image_past_sub), axis=0) 
            combined_viz_goal_image = torch.cat((viz_goal_image, viz_goal_image_sub), axis=0) 
            combined_goal_pos = torch.cat((goal_pos, goal_pos_sub), axis=0) 

            combined_local_yaw = torch.cat((local_yaw, torch.ones(Bsub)), axis=0)                                            
            combined_dist_pred, combined_action_pred = model(combined_obs_image, combined_goal_image)   
                                
            losses = _compute_losses(
                dist_label=combined_distance,
                action_label=combined_actions,
                dist_pred=combined_dist_pred,
                action_pred=combined_action_pred,
                alpha=0.5,
                learn_angle=True,
                action_mask=combined_action_mask,
            )            
            
            optimizer.zero_grad()
            losses["total_loss"].backward()
            optimizer.step()

            # Logging            
            loss_cpu = losses["total_loss"].item()
            dist_loss_cpu = losses["dist_loss"].item()
            action_loss_cpu = losses["action_loss"].item()                        
            tepoch.set_postfix(loss=loss_cpu)
            wandb.log({"total_loss": loss_cpu})
            wandb.log({"dist_loss": dist_loss_cpu})
            wandb.log({"action_loss": action_loss_cpu})            
            
            if epoch == 0 and i == 2000:
                lr_scheduler.step()
            
            if i % 5000 == 0 and i != 0:
                lr_scheduler.step()

            #if False:
            if i % 500 == 0 and i != 0:
                if no_emamodel:
                    numbered_path = os.path.join(project_folder, f"ema_{epoch}.pth")
                    torch.save(ema_model.averaged_model.state_dict(), numbered_path)

                numbered_path = os.path.join(project_folder, f"{epoch}.pth")
                torch.save(model.state_dict(), numbered_path)
                torch.save(model.state_dict(), latest_path)

                # save optimizer
                numbered_path = os.path.join(project_folder, f"optimizer_{epoch}.pth")
                latest_optimizer_path = os.path.join(project_folder, f"optimizer_latest.pth")
                torch.save(optimizer.state_dict(), latest_optimizer_path)

                # save scheduler
                numbered_path = os.path.join(project_folder, f"scheduler_{epoch}.pth")
                latest_scheduler_path = os.path.join(project_folder, f"scheduler_latest.pth")
                torch.save(lr_scheduler.state_dict(), latest_scheduler_path)
        
            #if False:
            if i % print_log_freq == 0:
                losses = {}
                losses['total_loss'] = loss_cpu
                losses['dist_loss'] = dist_loss_cpu
                losses['action_loss'] = action_loss_cpu             
                                                                
                for key, value in losses.items():
                    if key in loggers:
                        logger = loggers[key]
                        logger.log_data(value)
            
                data_log = {}
                for key, logger in loggers.items():
                    data_log[logger.full_name()] = logger.latest()
                    if i % print_log_freq == 0 and print_log_freq != 0:
                        print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")

                if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
                    wandb.log(data_log, commit=True)
            
            if image_log_freq != 0 and i % image_log_freq == 0:                
                visualize_il_estimation(
                    combined_viz_obs_image, 
                    combined_viz_obs_image_past,                     
                    combined_viz_goal_image,
                    combined_goal_pos,
                    combined_local_yaw,
                    combined_action_pred,
                    "train",
                    project_folder,
                    epoch,
                    num_images_log,
                    30,                    
                    use_wandb,
                    )               
                                     
###
def train_il2_dist_gnm_gps(
    model: nn.Module,
    model_GNM: nn.Module,    
    ema_model: EMAModel,
    optimizer: Adam,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    latest_path: str,
    dataloader: DataLoader,
    dataloader_sub: DataLoader,    
    transform: transforms,
    device: torch.device,
    project_folder: str,
    epoch: int,
    sacson: bool,
    no_emamodel: bool,      
    len_traj_pred: int,       
    alpha: float = 1e-4,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    use_wandb: bool = True,   
):
    """
    Train the model for one epoch.

    Args:
        model: model to train
        ema_model: exponential moving average model
        optimizer: optimizer to use
        dataloader: dataloader for training
        transform: transform to use
        device: device to use
        noise_scheduler: noise scheduler to train with 
        project_folder: folder to save images to
        epoch: current epoch
        alpha: weight of action loss
        print_log_freq: how often to print loss
        image_log_freq: how often to log images
        num_images_log: number of images to log
        use_wandb: whether to use wandb
    """
    #goal_mask_prob = torch.clip(torch.tensor(goal_mask_prob), 0, 1)
    model.train()
    num_batches = len(dataloader)

    total_loss_logger = Logger("total_loss", "train", window_size=print_log_freq)
    action_loss_logger = Logger("action_loss", "train", window_size=print_log_freq)    

    loggers = {
        "total_loss": total_loss_logger,
        "action_loss": action_loss_logger,                
    }     
    dataloader_sub_iter = iter(dataloader_sub)           
    
    model_GNM.eval().to(device)        
    with tqdm.tqdm(dataloader, desc="Train Batch", leave=False) as tepoch:
        for i, data in enumerate(tepoch):
            
            (
                obs_image, 
                goal_image,
                goal_image2,                
                current_image,                
                actions,
                distance,
                goal_pos,
                local_goal_mat,
                local_yaw,              
                actions_raw,
                obs_image_future,
                #_, 
                id_num,
                action_mask,
                ped_list,
                ped_list_raw,
                ped_list_no_trans,
                robot_list,
            ) = data
            
            try:
                (
                    obs_image_sub,
                    goal_image_sub,
                    action_label_sub,
                    dist_label_sub,
                    goal_pos_sub,
                    goal_yaw_sub,                    
                    dataset_index_sub,
                    action_mask_sub,
                    #_,
                    #_,
                ) = next(dataloader_sub_iter)
            except StopIteration:
                dataloader_sub_iter = iter(dataloader_sub) 
                (
                    obs_image_sub,
                    goal_image_sub,
                    action_label_sub,
                    dist_label_sub,
                    goal_pos_sub,
                    goal_yaw_sub,                     
                    dataset_index_sub,
                    action_mask_sub,
                    #_,
                    #_,
                ) = next(dataloader_sub_iter)        
    
            Bsub, _, H, W = obs_image_sub.size()  

            obs_images_sub = torch.split(obs_image_sub, 3, dim=1)
            viz_obs_image_sub = TF.resize(obs_images_sub[-1], VISUALIZATION_IMAGE_SIZE[::-1])
            viz_obs_image_past_sub = TF.resize(obs_images_sub[0], VISUALIZATION_IMAGE_SIZE[::-1])
            obs_images_sub = [transform(obs_image_sub).to(device) for obs_image_sub in obs_images_sub]
            obs_image_sub = torch.cat(obs_images_sub, dim=1)

            viz_goal_image_sub = TF.resize(goal_image_sub, VISUALIZATION_IMAGE_SIZE[::-1])        
            goal_image_sub = transform(goal_image_sub).to(device)
            dist_label_sub = dist_label_sub.to(device)
            action_label_sub = action_label_sub.to(device)
            action_mask_sub = action_mask_sub.to(device)
            goal_mask_sub = dist_label_sub > -1.0
            
            goal_pose_gps_sub = torch.cat((goal_pos_sub, torch.cos(goal_yaw_sub).unsqueeze(1), torch.sin(goal_yaw_sub).unsqueeze(1)), axis=1)
                    
            if psutil.virtual_memory().percent > 90.0:
                print("RAM usage (%)", psutil.virtual_memory().percent)
                break
            
            B, _, H, W = obs_image.size()              
            obs_images = torch.split(obs_image, 3, dim=1)
            viz_obs_image = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE[::-1])
            viz_obs_image_past = TF.resize(obs_images[0], VISUALIZATION_IMAGE_SIZE[::-1])     
            
            obs_images_future = torch.split(obs_image_future, 3, dim=1)                   
            obs_images = [transform(obs_image).to(device) for obs_image in obs_images]
            obs_image = torch.cat(obs_images, dim=1)
            actions = actions.to(device)
            action_mask = action_mask.to(device)

            batch_goal_pos = goal_pos.to(device)       
            goal_pose_gps = torch.cat((goal_pos, local_goal_mat[:,1,1].unsqueeze(1), local_goal_mat[:,1,0].unsqueeze(1)), axis=1)
                
            # Get distance label
            distance = distance.float().to(device)
            goal_mask = distance > 0.1                        

            for ig in range(B):
                if not goal_mask[ig]:
                    distance[ig] = 20
                    igr = random.randint(0, B-1) 
                    while ig == igr:
                        igr = random.randint(0, B-1) 
                    goal_image[ig] = goal_image[igr]

            viz_goal_image = TF.resize(goal_image, VISUALIZATION_IMAGE_SIZE[::-1])
            goal_image = transform(goal_image).to(device)         
            
            combined_obs_image = torch.cat((obs_image, obs_image_sub), axis=0)
            combined_goal_image = torch.cat((goal_image, goal_image_sub), axis=0)            
            combined_actions_origin = torch.cat((actions, action_label_sub), axis=0)   
             
            combined_distance = torch.cat((distance, dist_label_sub), axis=0)   
            combined_goal_mask = torch.cat((goal_mask, goal_mask_sub), axis=0)                   
            combined_action_mask = torch.cat((action_mask, action_mask_sub), axis=0) 

            combined_viz_obs_image = torch.cat((viz_obs_image, viz_obs_image_sub), axis=0)                   
            combined_viz_obs_image_past = torch.cat((viz_obs_image_past, viz_obs_image_past_sub), axis=0) 
            combined_viz_goal_image = torch.cat((viz_goal_image, viz_goal_image_sub), axis=0) 
            combined_goal_pos = torch.cat((goal_pos, goal_pos_sub), axis=0) 
            combined_goal_pos_gps = torch.cat((goal_pose_gps, goal_pose_gps_sub), axis=0).to(device) 
            
            combined_local_yaw = torch.cat((local_yaw, torch.ones(Bsub)), axis=0)                                            
            combined_action_pred = model(combined_obs_image, combined_goal_pos_gps)   

            #labeling by Robotic foundation model (IL on GNM dataset)
            with torch.no_grad():
                dist_estfrod, action_estfrod = model_GNM(obs_image, goal_image)                   
            combined_actions = torch.cat((action_estfrod.detach().to(device), action_label_sub), axis=0)    
                                                  
            losses = _compute_losses_gps(
                action_label=combined_actions_origin.to(device),                
                action_pred=combined_action_pred,
                learn_angle=True,
                action_mask=combined_action_mask,
            )            
            
            optimizer.zero_grad()
            losses["total_loss"].backward()
            optimizer.step()
            
            # Logging            
            loss_cpu = losses["total_loss"].item()
            action_loss_cpu = losses["action_loss"].item()                        
            tepoch.set_postfix(loss=loss_cpu)
            wandb.log({"total_loss": loss_cpu})
            wandb.log({"action_loss": action_loss_cpu})            
            
            if epoch == 0 and i == 2000:
                lr_scheduler.step()
            
            if i % 5000 == 0 and i != 0:
                lr_scheduler.step()

            if i % 500 == 0 and i != 0:
                if no_emamodel:
                    numbered_path = os.path.join(project_folder, f"ema_{epoch}.pth")
                    torch.save(ema_model.averaged_model.state_dict(), numbered_path)

                numbered_path = os.path.join(project_folder, f"{epoch}.pth")
                torch.save(model.state_dict(), numbered_path)
                torch.save(model.state_dict(), latest_path)

                # save optimizer
                numbered_path = os.path.join(project_folder, f"optimizer_{epoch}.pth")
                latest_optimizer_path = os.path.join(project_folder, f"optimizer_latest.pth")
                torch.save(optimizer.state_dict(), latest_optimizer_path)

                # save scheduler
                numbered_path = os.path.join(project_folder, f"scheduler_{epoch}.pth")
                latest_scheduler_path = os.path.join(project_folder, f"scheduler_latest.pth")
                torch.save(lr_scheduler.state_dict(), latest_scheduler_path)
        
            #if False:
            if i % print_log_freq == 0:
                losses = {}
                losses['total_loss'] = loss_cpu
                losses['action_loss'] = action_loss_cpu             
                                                                
                for key, value in losses.items():
                    if key in loggers:
                        logger = loggers[key]
                        logger.log_data(value)
            
                data_log = {}
                for key, logger in loggers.items():
                    data_log[logger.full_name()] = logger.latest()
                    if i % print_log_freq == 0 and print_log_freq != 0:
                        print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")

                if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
                    wandb.log(data_log, commit=True)
            
            if image_log_freq != 0 and i % image_log_freq == 0:                
                visualize_il2_estimation(
                    combined_viz_obs_image, 
                    combined_viz_obs_image_past,                     
                    combined_viz_goal_image,
                    combined_goal_pos,
                    combined_local_yaw,
                    combined_action_pred,
                    combined_actions,
                    combined_actions_origin,
                    "train",
                    project_folder,
                    epoch,
                    num_images_log,
                    30,                    
                    use_wandb,
                    )      
###
def train_il_exaug_dist_gnm_gps(
    model: nn.Module,
    model_GNM: nn.Module,    
    ema_model: EMAModel,
    optimizer: Adam,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    latest_path: str,
    dataloader: DataLoader,
    dataloader_sub: DataLoader,    
    transform: transforms,
    device: torch.device,
    project_folder: str,
    epoch: int,
    sacson: bool,
    no_emamodel: bool,  
    len_traj_pred: int,       
    alpha: float = 1e-4,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    use_wandb: bool = True,   
):
    """
    Train the model for one epoch.

    Args:
        model: model to train
        ema_model: exponential moving average model
        optimizer: optimizer to use
        dataloader: dataloader for training
        transform: transform to use
        device: device to use
        noise_scheduler: noise scheduler to train with 
        project_folder: folder to save images to
        epoch: current epoch
        alpha: weight of action loss
        print_log_freq: how often to print loss
        image_log_freq: how often to log images
        num_images_log: number of images to log
        use_wandb: whether to use wandb
    """
    model.train()
    num_batches = len(dataloader)

    total_loss_logger = Logger("total_loss", "train", window_size=print_log_freq)
    action_loss_logger = Logger("action_loss", "train", window_size=print_log_freq)    

    loggers = {
        "total_loss": total_loss_logger,
        "action_loss": action_loss_logger,                
    }          
    dataloader_sub_iter = iter(dataloader_sub)           
    
    model_GNM.eval().to(device)          
    with tqdm.tqdm(dataloader, desc="Train Batch", leave=False) as tepoch:
        for i, data in enumerate(tepoch):
            
            (
                obs_image, 
                goal_image,
                goal_image2,                
                current_image,                
                actions,
                distance,
                goal_pos,
                local_goal_mat,
                local_yaw,              
                actions_raw,
                obs_image_future,
                #_, 
                id_num,
                action_mask,
                ped_list,
                ped_list_raw,
                ped_list_no_trans,
                robot_list,
            ) = data

            try:
                (
                    obs_image_sub,
                    goal_image_sub,
                    action_label_sub,
                    dist_label_sub,
                    goal_pos_sub,
                    goal_yaw_sub,                    
                    dataset_index_sub,
                    action_mask_sub,
                    #_,
                    #_,
                ) = next(dataloader_sub_iter)
            except StopIteration:
                dataloader_sub_iter = iter(dataloader_sub) 
                (
                    obs_image_sub,
                    goal_image_sub,
                    action_label_sub,
                    dist_label_sub,
                    goal_pos_sub,
                    goal_yaw_sub,                     
                    dataset_index_sub,
                    action_mask_sub,
                    #_,
                    #_,
                ) = next(dataloader_sub_iter)        
    
            Bsub, _, H, W = obs_image_sub.size()  

            obs_images_sub = torch.split(obs_image_sub, 3, dim=1)
            viz_obs_image_sub = TF.resize(obs_images_sub[-1], VISUALIZATION_IMAGE_SIZE[::-1])
            viz_obs_image_past_sub = TF.resize(obs_images_sub[0], VISUALIZATION_IMAGE_SIZE[::-1])
            obs_images_sub = [transform(obs_image_sub).to(device) for obs_image_sub in obs_images_sub]
            obs_image_sub = torch.cat(obs_images_sub, dim=1)

            viz_goal_image_sub = TF.resize(goal_image_sub, VISUALIZATION_IMAGE_SIZE[::-1])
            goal_image_sub = transform(goal_image_sub).to(device)

            dist_label_sub = dist_label_sub.to(device)
            action_label_sub = action_label_sub.to(device)
            action_mask_sub = action_mask_sub.to(device)
            goal_mask_sub = dist_label_sub > -1.0
            
            goal_pose_gps_sub = torch.cat((goal_pos_sub, torch.cos(goal_yaw_sub).unsqueeze(1), torch.sin(goal_yaw_sub).unsqueeze(1)), axis=1)

            if psutil.virtual_memory().percent > 90.0:
                print("RAM usage (%)", psutil.virtual_memory().percent)
                break
            
            B, _, H, W = obs_image.size()  
            obs_images = torch.split(obs_image, 3, dim=1)
            viz_obs_image = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE[::-1])
            viz_obs_image_past = TF.resize(obs_images[0], VISUALIZATION_IMAGE_SIZE[::-1])     
            
            obs_images_future = torch.split(obs_image_future, 3, dim=1)
                   
            obs_images = [transform(obs_image).to(device) for obs_image in obs_images]
            obs_image = torch.cat(obs_images, dim=1)
            actions = actions.to(device)
            action_mask = action_mask.to(device)

            batch_goal_pos = goal_pos.to(device)    
            goal_pose_gps = torch.cat((goal_pos, local_goal_mat[:,1,1].unsqueeze(1), local_goal_mat[:,1,0].unsqueeze(1)), axis=1)
                
            # Get distance label
            distance = distance.float().to(device)
            goal_mask = distance > 0.1                        

            for ig in range(B):
                if not goal_mask[ig]:
                    distance[ig] = 20
                    igr = random.randint(0, B-1) 
                    while ig == igr:
                        igr = random.randint(0, B-1) 
                    goal_image[ig] = goal_image[igr]

            viz_goal_image = TF.resize(goal_image, VISUALIZATION_IMAGE_SIZE[::-1])
            goal_image = transform(goal_image).to(device)         
            goal_image2 = transform(goal_image2).to(device)         
                        
            combined_obs_image = torch.cat((obs_image, obs_image_sub), axis=0)
            combined_goal_image = torch.cat((goal_image, goal_image_sub), axis=0)            
            combined_actions_origin = torch.cat((actions, action_label_sub), axis=0)   
            
            combined_distance = torch.cat((distance, dist_label_sub), axis=0)   
            combined_goal_mask = torch.cat((goal_mask, goal_mask_sub), axis=0)                   
            combined_action_mask = torch.cat((action_mask, action_mask_sub), axis=0) 

            combined_viz_obs_image = torch.cat((viz_obs_image, viz_obs_image_sub), axis=0)                   
            combined_viz_obs_image_past = torch.cat((viz_obs_image_past, viz_obs_image_past_sub), axis=0) 
            combined_viz_goal_image = torch.cat((viz_goal_image, viz_goal_image_sub), axis=0) 
            combined_goal_pos = torch.cat((goal_pos, goal_pos_sub), axis=0) 
            combined_goal_pos_gps = torch.cat((goal_pose_gps, goal_pose_gps_sub), axis=0).to(device) 
            
            combined_local_yaw = torch.cat((local_yaw, torch.ones(Bsub)), axis=0)                                            
            combined_action_pred = model(combined_obs_image, combined_goal_pos_gps)   

            rsize = 0.3*torch.ones(B, 1, 1).to(device) #robot radius : 0 -- 1.0 m
            delay = torch.zeros(B, 1, 1).to(device)   
            linear_vel_old = 0.5*torch.ones(B, 6).float().to(device)
            angular_vel_old = 0.0*torch.ones(B, 6).float().to(device)
            vel_past = torch.cat((linear_vel_old, angular_vel_old), axis=1).unsqueeze(2)          
             
            with torch.no_grad():
                linear_vel, angular_vel, dist_estfrod = model_GNM(obs_image, goal_image2, rsize, delay, vel_past)
                                                
            linear_vel_d = linear_vel
            angular_vel_d = angular_vel            
                
            px_ref_list, pz_ref_list, ry_ref_list = robot_pos_model_fix(linear_vel_d, angular_vel_d)         
            
            
            x_traj = []
            z_traj = []
            yaw_traj = [] 
            for ic in range(len(px_ref_list)):
                x_traj.append(px_ref_list[ic].unsqueeze(1))
                z_traj.append(pz_ref_list[ic].unsqueeze(1))
                yaw_traj.append(ry_ref_list[ic].unsqueeze(1))                            
            x_traj_cat = torch.cat(x_traj, axis = 1)
            z_traj_cat = torch.cat(z_traj, axis = 1)
            yaw_traj_cat = torch.cat(yaw_traj, axis = 1)                        
            
            metric_waypoint_spacing = 0.25*0.5
            action_estfrod = torch.cat((z_traj_cat.unsqueeze(-1)/metric_waypoint_spacing, -x_traj_cat.unsqueeze(-1)/metric_waypoint_spacing, torch.cos(-yaw_traj_cat).unsqueeze(-1), torch.sin(-yaw_traj_cat).unsqueeze(-1)), axis=2)     
            combined_actions = torch.cat((action_estfrod.detach().to(device), action_label_sub), axis=0)   
                                                  
            losses = _compute_losses_gps(
                action_label=combined_actions.to(device),
                action_pred=combined_action_pred,
                learn_angle=True,
                action_mask=combined_action_mask,
            )            
            
            optimizer.zero_grad()
            losses["total_loss"].backward()
            optimizer.step()
            
            # Logging            
            loss_cpu = losses["total_loss"].item()
            action_loss_cpu = losses["action_loss"].item()                        
            tepoch.set_postfix(loss=loss_cpu)
            wandb.log({"total_loss": loss_cpu})
            wandb.log({"action_loss": action_loss_cpu})            
            
            if epoch == 0 and i == 2000:
                lr_scheduler.step()
            
            if i % 5000 == 0 and i != 0:
                lr_scheduler.step()

            if i % 500 == 0 and i != 0:
                if no_emamodel:
                    numbered_path = os.path.join(project_folder, f"ema_{epoch}.pth")
                    torch.save(ema_model.averaged_model.state_dict(), numbered_path)

                numbered_path = os.path.join(project_folder, f"{epoch}.pth")
                torch.save(model.state_dict(), numbered_path)
                torch.save(model.state_dict(), latest_path)

                # save optimizer
                numbered_path = os.path.join(project_folder, f"optimizer_{epoch}.pth")
                latest_optimizer_path = os.path.join(project_folder, f"optimizer_latest.pth")
                torch.save(optimizer.state_dict(), latest_optimizer_path)

                # save scheduler
                numbered_path = os.path.join(project_folder, f"scheduler_{epoch}.pth")
                latest_scheduler_path = os.path.join(project_folder, f"scheduler_latest.pth")
                torch.save(lr_scheduler.state_dict(), latest_scheduler_path)
        
            if i % print_log_freq == 0:
                losses = {}
                losses['total_loss'] = loss_cpu
                losses['action_loss'] = action_loss_cpu             
                                                                
                for key, value in losses.items():
                    if key in loggers:
                        logger = loggers[key]
                        logger.log_data(value)
            
                data_log = {}
                for key, logger in loggers.items():
                    data_log[logger.full_name()] = logger.latest()
                    if i % print_log_freq == 0 and print_log_freq != 0:
                        print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")

                if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
                    wandb.log(data_log, commit=True)
            
            if image_log_freq != 0 and i % image_log_freq == 0:                
                visualize_il2_estimation(
                    combined_viz_obs_image, 
                    combined_viz_obs_image_past,                     
                    combined_viz_goal_image,
                    combined_goal_pos,
                    combined_local_yaw,
                    combined_action_pred,
                    combined_actions,
                    combined_actions_origin,
                    "train",
                    project_folder,
                    epoch,
                    num_images_log,
                    30,                    
                    use_wandb,
                    )      
###
def train_il_exaug_dist_gnm_gps_map(
    model: nn.Module,
    model_GNM: nn.Module,    
    ema_model: EMAModel,
    optimizer: Adam,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    latest_path: str,
    dataloader: DataLoader,
    dataloader_sub: DataLoader,    
    transform: transforms,
    device: torch.device,
    project_folder: str,
    epoch: int,
    sacson: bool,
    no_emamodel: bool, 
    len_traj_pred: int,       
    alpha: float = 1e-4,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    use_wandb: bool = True,   
):
    """
    Train the model for one epoch.

    Args:
        model: model to train
        ema_model: exponential moving average model
        optimizer: optimizer to use
        dataloader: dataloader for training
        transform: transform to use
        device: device to use
        noise_scheduler: noise scheduler to train with 
        project_folder: folder to save images to
        epoch: current epoch
        alpha: weight of action loss
        print_log_freq: how often to print loss
        image_log_freq: how often to log images
        num_images_log: number of images to log
        use_wandb: whether to use wandb
    """
    model.train()
    num_batches = len(dataloader)

    total_loss_logger = Logger("total_loss", "train", window_size=print_log_freq)
    action_loss_logger = Logger("action_loss", "train", window_size=print_log_freq)    

    loggers = {
        "total_loss": total_loss_logger,
        "action_loss": action_loss_logger,                
    }
    
    D = np.array(np.load('/media/noriaki/Noriaki_Data/learning-language-navigation_SACSoN/train/vint_train/training/fisheye_calibration/D.npy', mmap_mode='r'))
    K = np.array(np.load('/media/noriaki/Noriaki_Data/learning-language-navigation_SACSoN/train/vint_train/training/fisheye_calibration/K.npy', mmap_mode='r'))

    mask_360 = np.loadtxt(open("/media/noriaki/Noriaki_Data/learning-language-navigation_SACSoN/train/mask_360view.csv", "rb"), delimiter=",", skiprows=0)   
    mask_360_resize = np.repeat(np.expand_dims(cv2.resize(mask_360, (832, 128)), 0), 3, 0).astype(np.float32)
    mask_360_torch = torch.from_numpy(mask_360_resize[:,:,0:416]).unsqueeze(0).to(device)
               
    dataloader_sub_iter = iter(dataloader_sub)           
    
    model_GNM.eval().to(device)      
    with tqdm.tqdm(dataloader, desc="Train Batch", leave=False) as tepoch:
        for i, data in enumerate(tepoch):
            
            (
                obs_image, 
                goal_image,
                goal_image2,                
                current_image,                
                actions,
                distance,
                goal_pos,
                local_goal_mat,
                local_yaw,              
                actions_raw,
                obs_image_future,
                #_, 
                id_num,
                action_mask,
                current_map_image,
                goal_map_image,
                ped_list_no_trans,
                robot_list,
            ) = data
            
            try:
                (
                    obs_image_sub,
                    goal_image_sub,
                    action_label_sub,
                    dist_label_sub,
                    goal_pos_sub,
                    goal_yaw_sub,                    
                    dataset_index_sub,
                    action_mask_sub,
                    #_,
                    #_,
                ) = next(dataloader_sub_iter)
            except StopIteration:
                dataloader_sub_iter = iter(dataloader_sub) 
                (
                    obs_image_sub,
                    goal_image_sub,
                    action_label_sub,
                    dist_label_sub,
                    goal_pos_sub,
                    goal_yaw_sub,                     
                    dataset_index_sub,
                    action_mask_sub,
                    #_,
                    #_,
                ) = next(dataloader_sub_iter)           
            Bsub, _, H, W = obs_image_sub.size()  

            current_map_image_sub = torch.zeros(Bsub, 3, 96, 96)
            goal_map_image_sub = torch.zeros(Bsub, 3, 96, 96)

            obs_images_sub = torch.split(obs_image_sub, 3, dim=1)
            viz_obs_image_sub = TF.resize(obs_images_sub[-1], VISUALIZATION_IMAGE_SIZE[::-1])
            viz_obs_image_past_sub = TF.resize(obs_images_sub[0], VISUALIZATION_IMAGE_SIZE[::-1])
            obs_images_sub = [transform(obs_image_sub).to(device) for obs_image_sub in obs_images_sub]
            obs_image_sub_map = obs_images_sub[-1].to(device)
            obs_image_sub = torch.cat(obs_images_sub, dim=1)

            map_images_sub = torch.cat((transform(current_map_image_sub).to(device), transform(goal_map_image_sub).to(device), obs_image_sub_map), axis=1)
            viz_goal_image_sub = TF.resize(goal_image_sub, VISUALIZATION_IMAGE_SIZE[::-1])
        
            goal_image_sub = transform(goal_image_sub).to(device)

            dist_label_sub = dist_label_sub.to(device)
            action_label_sub = action_label_sub.to(device)
            action_mask_sub = action_mask_sub.to(device)
            goal_mask_sub = dist_label_sub > -1.0
            
            goal_pose_gps_sub = torch.cat((goal_pos_sub, torch.cos(goal_yaw_sub).unsqueeze(1), torch.sin(goal_yaw_sub).unsqueeze(1)), axis=1)
                    
            if psutil.virtual_memory().percent > 90.0:
                print("RAM usage (%)", psutil.virtual_memory().percent)
                break
            
            B, _, H, W = obs_image.size()  
            
            obs_images = torch.split(obs_image, 3, dim=1)
            viz_obs_image = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE[::-1])
            viz_obs_image_past = TF.resize(obs_images[0], VISUALIZATION_IMAGE_SIZE[::-1])                 
            obs_images_future = torch.split(obs_image_future, 3, dim=1)
                   
            obs_images = [transform(obs_image).to(device) for obs_image in obs_images]
            obs_image_map = obs_images[-1].to(device)
            map_images = torch.cat((transform(current_map_image).to(device), transform(goal_map_image).to(device), obs_image_map), axis=1)            
            
            obs_image = torch.cat(obs_images, dim=1)
            actions = actions.to(device)
            action_mask = action_mask.to(device)
            batch_goal_pos = goal_pos.to(device)        
            goal_pose_gps = torch.cat((goal_pos, local_goal_mat[:,1,1].unsqueeze(1), local_goal_mat[:,1,0].unsqueeze(1)), axis=1)
                
            # Get distance label
            distance = distance.float().to(device)
            goal_mask = distance > 0.1                        

            for ig in range(B):
                if not goal_mask[ig]:
                    distance[ig] = 20
                    igr = random.randint(0, B-1) 
                    while ig == igr:
                        igr = random.randint(0, B-1) 
                    goal_image[ig] = goal_image[igr]

            viz_goal_image = TF.resize(goal_image, VISUALIZATION_IMAGE_SIZE[::-1])
            goal_image = transform(goal_image).to(device)         
            goal_image2 = transform(goal_image2).to(device)         
                        
            combined_obs_image = torch.cat((obs_image, obs_image_sub), axis=0)
            combined_goal_image = torch.cat((goal_image, goal_image_sub), axis=0)            
            combined_actions_origin = torch.cat((actions, action_label_sub), axis=0)            
            combined_distance = torch.cat((distance, dist_label_sub), axis=0)   
            combined_goal_mask = torch.cat((goal_mask, goal_mask_sub), axis=0)                   
            combined_action_mask = torch.cat((action_mask, action_mask_sub), axis=0) 
            combined_viz_obs_image = torch.cat((viz_obs_image, viz_obs_image_sub), axis=0)                   
            combined_viz_obs_image_past = torch.cat((viz_obs_image_past, viz_obs_image_past_sub), axis=0)          
            combined_viz_cur_map = torch.cat((current_map_image, current_map_image_sub), axis=0)
            combined_viz_goal_map = torch.cat((goal_map_image, goal_map_image_sub), axis=0)                   
            combined_viz_goal_image = torch.cat((viz_goal_image, viz_goal_image_sub), axis=0) 
            combined_goal_pos = torch.cat((goal_pos, goal_pos_sub), axis=0) 
            combined_goal_pos_gps = torch.cat((goal_pose_gps, goal_pose_gps_sub), axis=0).to(device) 
            combined_map_images = torch.cat((map_images, map_images_sub), axis=0)
            combined_local_yaw = torch.cat((local_yaw, torch.ones(Bsub)), axis=0)     
            
            #model calculate 
            goal_mask_f = torch.randint(0,3,(B,)).to(device) #Frodobot: Goal pose, or Satellite or (Goal pose + Satellite)                   
            goal_mask_sub = torch.zeros(Bsub).to(device) #GNM: Goal pose only
            goal_mask_select = torch.cat((goal_mask_f, goal_mask_sub), axis=0)
            
            combined_action_pred, mask_number = model(combined_obs_image, combined_goal_pos_gps, combined_map_images, goal_mask_select)   

            rsize = 0.3*torch.ones(B, 1, 1).to(device) #robot radius : 0 -- 1.0 m
            delay = torch.zeros(B, 1, 1).to(device)   
            linear_vel_old = 0.5*torch.ones(B, 6).float().to(device)
            angular_vel_old = 0.0*torch.ones(B, 6).float().to(device)
            vel_past = torch.cat((linear_vel_old, angular_vel_old), axis=1).unsqueeze(2)          
             
            with torch.no_grad():
                linear_vel, angular_vel, dist_estfrod = model_GNM(obs_image, goal_image2, rsize, delay, vel_past)
                                                
            linear_vel_d = linear_vel
            angular_vel_d = angular_vel            
                
            px_ref_list, pz_ref_list, ry_ref_list = robot_pos_model_fix(linear_vel_d, angular_vel_d)         
                     
            x_traj = []
            z_traj = []
            yaw_traj = [] 
            for ic in range(len(px_ref_list)):
                x_traj.append(px_ref_list[ic].unsqueeze(1))
                z_traj.append(pz_ref_list[ic].unsqueeze(1))
                yaw_traj.append(ry_ref_list[ic].unsqueeze(1))                            
            x_traj_cat = torch.cat(x_traj, axis = 1)
            z_traj_cat = torch.cat(z_traj, axis = 1)
            yaw_traj_cat = torch.cat(yaw_traj, axis = 1)                        
            
            metric_waypoint_spacing = 0.25*0.5
            action_estfrod = torch.cat((z_traj_cat.unsqueeze(-1)/metric_waypoint_spacing, -x_traj_cat.unsqueeze(-1)/metric_waypoint_spacing, torch.cos(-yaw_traj_cat).unsqueeze(-1), torch.sin(-yaw_traj_cat).unsqueeze(-1)), axis=2)     
            combined_actions = torch.cat((action_estfrod.detach().to(device), action_label_sub), axis=0)   
                                                  
            losses = _compute_losses_gps(
                action_label=combined_actions.to(device),
                action_pred=combined_action_pred,
                learn_angle=True,
                action_mask=combined_action_mask,
            )            
            
            optimizer.zero_grad()
            losses["total_loss"].backward()
            optimizer.step()
            
            # Logging            
            loss_cpu = losses["total_loss"].item()
            action_loss_cpu = losses["action_loss"].item()                        
            tepoch.set_postfix(loss=loss_cpu)
            wandb.log({"total_loss": loss_cpu})
            wandb.log({"action_loss": action_loss_cpu})            
            
            if epoch == 0 and i == 2000:
                lr_scheduler.step()
            
            if i % 5000 == 0 and i != 0:
                lr_scheduler.step()

            #if False:
            if i % 500 == 0 and i != 0:
                if no_emamodel:
                    numbered_path = os.path.join(project_folder, f"ema_{epoch}.pth")
                    torch.save(ema_model.averaged_model.state_dict(), numbered_path)

                numbered_path = os.path.join(project_folder, f"{epoch}.pth")
                torch.save(model.state_dict(), numbered_path)
                torch.save(model.state_dict(), latest_path)

                # save optimizer
                numbered_path = os.path.join(project_folder, f"optimizer_{epoch}.pth")
                latest_optimizer_path = os.path.join(project_folder, f"optimizer_latest.pth")
                torch.save(optimizer.state_dict(), latest_optimizer_path)

                # save scheduler
                numbered_path = os.path.join(project_folder, f"scheduler_{epoch}.pth")
                latest_scheduler_path = os.path.join(project_folder, f"scheduler_latest.pth")
                torch.save(lr_scheduler.state_dict(), latest_scheduler_path)
        
            #if False:
            if i % print_log_freq == 0:
                losses = {}
                losses['total_loss'] = loss_cpu
                losses['action_loss'] = action_loss_cpu             
                                                                
                for key, value in losses.items():
                    if key in loggers:
                        logger = loggers[key]
                        logger.log_data(value)
            
                data_log = {}
                for key, logger in loggers.items():
                    data_log[logger.full_name()] = logger.latest()
                    if i % print_log_freq == 0 and print_log_freq != 0:
                        print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")

                if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
                    wandb.log(data_log, commit=True)
            
            if image_log_freq != 0 and i % image_log_freq == 0:                
                visualize_il2_estimation_map(
                    combined_viz_obs_image, 
                    combined_viz_obs_image_past,                     
                    combined_viz_goal_image,
                    combined_viz_cur_map,
                    combined_viz_goal_map,
                    combined_goal_pos,
                    combined_local_yaw,
                    combined_action_pred,
                    combined_actions,
                    combined_actions_origin,
                    mask_number,
                    "train",
                    project_folder,
                    epoch,
                    num_images_log,
                    30,                    
                    use_wandb,
                    )      

def evaluate_lelan(
    eval_type: str,
    ema_model: EMAModel,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    project_folder: str,
    epoch: int,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    eval_fraction: float = 0.25,
    use_wandb: bool = True,
):
    
    """
    Evaluate the model on the given evaluation dataset.

    Args:
        eval_type (string): f"{data_type}_{eval_type}" (e.g. "recon_train", "gs_test", etc.)
        ema_model (nn.Module): exponential moving average version of model to evaluate
        dataloader (DataLoader): dataloader for eval
        transform (transforms): transform to apply to images
        device (torch.device): device to use for evaluation
        project_folder (string): path to project folder
        epoch (int): current epoch    total_loss_logger = Logger("total loss", "train", window_size=print_log_freq)    
    """
    ema_model.eval()    
    num_batches = len(dataloader)

    total_loss_logger = Logger("total loss", eval_type, window_size=print_log_freq)    
    pose_loss_logger = Logger("pose loss", eval_type, window_size=print_log_freq)
    smooth_loss_logger = Logger("smooth loss", eval_type, window_size=print_log_freq)     
    loggers = {
        "total loss": total_loss_logger,    
        "pose loss": pose_loss_logger,
        "vel smooth loss": smooth_loss_logger,
    }    
    num_batches = max(int(num_batches * eval_fraction), 1)

    all_total = 0.0
    all_dist = 0.0
    all_diff = 0.0
    
    count_batch = 0
    data_size = 0
    with tqdm.tqdm(
        itertools.islice(dataloader, num_batches), 
        total=num_batches, 
        dynamic_ncols=True, 
        desc=f"Evaluating {eval_type} for epoch {epoch}", 
        leave=False) as tepoch:
        for i, data in enumerate(tepoch):
            (
                obs_images, 
                goal_image,
                obj_poses,
                obj_inst,
                goal_pos_norm,
            ) = data
            
            obs_images_list = torch.split(obs_images, 3, dim=1)
            obs_image = obs_images_list[-1]       

            batch_viz_obs_images = TF.resize((255.0*obs_image).type(torch.uint8), VISUALIZATION_IMAGE_SIZE[::-1])
            batch_viz_goal_images = TF.resize((255.0*goal_image).type(torch.uint8), VISUALIZATION_IMAGE_SIZE[::-1])                         
            batch_obs_images = transform(obs_image).to(device)
            batch_obj_poses = obj_poses.to(device)
            
            B = batch_obs_images.shape[0]
            with torch.no_grad():
                batch_obj_inst = clip.tokenize(obj_inst, truncate=True).to(device)          
                feat_text = ema_model("text_encoder", inst_ref=batch_obj_inst)                  
                obsgoal_cond = ema_model("vision_encoder", obs_img=batch_obs_images, feat_text = feat_text.to(dtype=torch.float32))
                linear_vel, angular_vel = ema_model("dist_pred_net", obsgoal_cond=obsgoal_cond)
                
                px_ref_list, pz_ref_list, ry_ref_list = robot_pos_model_fix(linear_vel, angular_vel)
                px_ref = px_ref_list[-1]
                pz_ref = pz_ref_list[-1]
                ry_ref = ry_ref_list[-1]
                                                    
            last_poses = torch.cat((px_ref.unsqueeze(1), pz_ref.unsqueeze(1)), axis=1)
                        
            dist_loss = nn.functional.mse_loss(last_poses, batch_obj_poses)   
            diff_loss = nn.functional.mse_loss(linear_vel[:,:-1], linear_vel[:,1:]) + nn.functional.mse_loss(angular_vel[:,:-1], angular_vel[:,1:]) 
                        
            # Logging
            loss_cpu = dist_loss.item()
            tepoch.set_postfix(loss=loss_cpu)

            wandb.log({"total_eval_loss": (dist_loss + 1.0*diff_loss).item()})
            wandb.log({"dist_eval_loss": dist_loss.item()})
            wandb.log({"diff_eval_loss": diff_loss.item()})

            all_total += (dist_loss + 1.0*diff_loss).item()
            all_dist += dist_loss.item()
            all_diff += diff_loss.item()
            count_batch += 1.0
            data_size += B
            if i % print_log_freq == 0 and print_log_freq != 0: 
                losses = {}
                losses['total loss'] = loss_cpu
                losses['pose loss'] = dist_loss.item()
                losses['vel smooth loss'] = diff_loss.item()             
                                                
                for key, value in losses.items():
                    if key in loggers:
                        logger = loggers[key]
                        logger.log_data(value)
            
                data_log = {}
                for key, logger in loggers.items():
                    data_log[logger.full_name()] = logger.latest()
                    if i % print_log_freq == 0 and print_log_freq != 0:
                        print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")

                if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
                    wandb.log(data_log, commit=True)
            
            if image_log_freq != 0 and i % image_log_freq == 0:
                visualize_lelan_estimation(
                    batch_viz_obs_images,
                    batch_viz_goal_images,
                    obj_poses,
                    obj_inst,
                    linear_vel.cpu(),
                    angular_vel.cpu(),
                    last_poses.cpu(),
                    eval_type,
                    project_folder,
                    epoch,
                    num_images_log,
                    30,                    
                    use_wandb,
                )                
    print(eval_type, "total loss:", all_total/count_batch, "dist loss:", all_dist/count_batch, "diff loss:", all_diff/count_batch, "batch count:", count_batch, "data size:", data_size)

def evaluate_lelan_col(
    eval_type: str,
    ema_model: EMAModel,
    ema_model_nomad: EMAModel,    
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    noise_scheduler: DDPMScheduler,
    project_folder: str,
    weight_col_loss: float,    
    epoch: int,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    eval_fraction: float = 0.25,
    use_wandb: bool = True,
):
    
    """
    Evaluate the model on the given evaluation dataset.

    Args:
        eval_type (string): f"{data_type}_{eval_type}" (e.g. "recon_train", "gs_test", etc.)
        ema_model (nn.Module): exponential moving average version of model to evaluate
        ema_model_nomad (nn.Module): exponential moving average version of pre-trained NoMaD policy
        dataloader (DataLoader): dataloader for eval
        transform (transforms): transform to apply to images
        device (torch.device): device to use for evaluation
        noise_scheduler: noise scheduler to evaluate with 
        project_folder (string): path to project folder
        weight_col_loss (float) : weight for collision avoidance loss 
        epoch (int): current epoch    total_loss_logger = Logger("total loss", "train", window_size=print_log_freq)    
    """

    ema_model.eval()
    ema_model_nomad = ema_model_nomad.averaged_model
    ema_model_nomad.eval()       
    num_batches = len(dataloader)

    total_loss_logger = Logger("total loss", eval_type, window_size=print_log_freq)    
    pose_loss_logger = Logger("pose loss", eval_type, window_size=print_log_freq)
    smooth_loss_logger = Logger("smooth loss", eval_type, window_size=print_log_freq)    
    col_loss_logger = Logger("col loss", eval_type, window_size=print_log_freq) 
    loggers = {
        "total loss": total_loss_logger,    
        "pose loss": pose_loss_logger,
        "vel smooth loss": smooth_loss_logger,
        "col loss": col_loss_logger,        
    }    
    num_batches = max(int(num_batches * eval_fraction), 1)

    all_total = 0.0
    all_dist = 0.0
    all_diff = 0.0
    all_col = 0.0
        
    count_batch = 0
    data_size = 0
    with tqdm.tqdm(
        itertools.islice(dataloader, num_batches), 
        total=num_batches, 
        dynamic_ncols=True, 
        desc=f"Evaluating {eval_type} for epoch {epoch}", 
        leave=False) as tepoch:
        for i, data in enumerate(tepoch):
            (
                obs_images, 
                goal_image,
                goal_pos,
                obj_inst,
                goal_pos_norm,
            ) = data
            
            obs_images_list = torch.split(obs_images, 3, dim=1)
            obs_image = obs_images_list[-1]              
            
            batch_viz_obs_images = TF.resize((255.0*obs_image).type(torch.uint8), VISUALIZATION_IMAGE_SIZE[::-1])
            batch_viz_goal_images = TF.resize((255.0*goal_image).type(torch.uint8), VISUALIZATION_IMAGE_SIZE[::-1])                                                       
            batch_obs_current = transform(obs_image).to(device)
            batch_goal_pos = goal_pos.to(device)
            goal_pos_norm = goal_pos_norm.to(device)                              
            batch_obs_images = [transform(TF.resize(obs, (96, 96), antialias=True)) for obs in obs_images_list]
            batch_obs_images = torch.cat(batch_obs_images, dim=1).to(device)
            batch_goal_images = transform(TF.resize(goal_image, (96, 96), antialias=True)).to(device)
            
            B = batch_obs_images.shape[0]
                        
            # split into batches
            batch_obs_images_list = torch.split(batch_obs_images, B, dim=0)
            batch_goal_images_list = torch.split(batch_goal_images, B, dim=0)

            with torch.no_grad():
                select_traj = supervision_from_nomad(
                    ema_model_nomad,
                    noise_scheduler,
                    batch_obs_images,
                    batch_goal_images,
                    batch_viz_obs_images,
                    batch_viz_goal_images,
                    goal_pos_norm,
                    device,
                    project_folder,
                    epoch,
                    B,
                    i,                
                    30,
                    use_wandb,
                    )                
            
            with torch.no_grad():
                batch_obj_inst = clip.tokenize(obj_inst, truncate=True).to(device)         
                feat_text = ema_model("text_encoder", inst_ref=batch_obj_inst)       
                                
                B = batch_obs_images.shape[0]

                obsgoal_cond = ema_model("vision_encoder", obs_img=batch_obs_images, feat_text = feat_text.to(dtype=torch.float32), current_img=batch_obs_current)
                linear_vel, angular_vel = ema_model("dist_pred_net", obsgoal_cond=obsgoal_cond)

                px_ref_list, pz_ref_list, ry_ref_list = robot_pos_model_fix(linear_vel, angular_vel)
                px_ref = px_ref_list[-1]
                pz_ref = pz_ref_list[-1]
                ry_ref = ry_ref_list[-1]
                                                    
            last_poses = torch.cat((px_ref.unsqueeze(1), pz_ref.unsqueeze(1)), axis=1)
            px_ref_listx = []
            pz_ref_listx = []
            for it in range(8):            
                px_ref_listx.append(px_ref_list[it].unsqueeze(1).unsqueeze(2))
                pz_ref_listx.append(pz_ref_list[it].unsqueeze(1).unsqueeze(2))
            traj_policy = torch.concat((torch.concat(pz_ref_listx, axis=1), -torch.concat(px_ref_listx, axis=1)), axis=2)
                                                
            dist_loss = nn.functional.mse_loss(last_poses, batch_goal_pos)   
            diff_loss = nn.functional.mse_loss(linear_vel[:,:-1], linear_vel[:,1:]) + nn.functional.mse_loss(angular_vel[:,:-1], angular_vel[:,1:]) 

            mask_nomad = (batch_goal_pos[:,1:2] > 1.0).float().unsqueeze(1).repeat(1,8,2)
            mask_dist = (~(batch_goal_pos[:,1:2] > 1.0)).float()
            sum_dist = mask_dist.sum()            
            col_loss = nn.functional.mse_loss(mask_nomad*traj_policy, 0.12*mask_nomad*select_traj)*float(B)/(float(B) - sum_dist.float() + 1e-7) #0.12 is de-normalization
            
            loss = 1.0*dist_loss + 1.0*diff_loss + weight_col_loss*col_loss
                                                
            # Logging
            loss_cpu = loss.item()
            tepoch.set_postfix(loss=loss_cpu)

            wandb.log({"total_eval_loss": (dist_loss + 1.0*diff_loss + weight_col_loss*col_loss).item()})
            wandb.log({"dist_eval_loss": dist_loss.item()})
            wandb.log({"diff_eval_loss": diff_loss.item()})
            wandb.log({"col_eval_loss": col_loss.item()})
            
            all_total += (dist_loss + 1.0*diff_loss + weight_col_loss*col_loss).item()
            all_dist += dist_loss.item()
            all_diff += diff_loss.item()
            all_col += col_loss.item()            
            count_batch += 1.0
            data_size += B
            if i % print_log_freq == 0 and print_log_freq != 0:
                losses = {}
                losses['total loss'] = loss_cpu
                losses['pose loss'] = dist_loss.item()
                losses['vel smooth loss'] = diff_loss.item()             
                losses['col loss'] = col_loss.item()       
                                                                
                for key, value in losses.items():
                    if key in loggers:
                        logger = loggers[key]
                        logger.log_data(value)
            
                data_log = {}
                for key, logger in loggers.items():
                    data_log[logger.full_name()] = logger.latest()
                    if i % print_log_freq == 0 and print_log_freq != 0:
                        print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")

                if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
                    wandb.log(data_log, commit=True)
            
            if image_log_freq != 0 and i % image_log_freq == 0:
                visualize_lelan_col_estimation(
                    batch_viz_obs_images,
                    batch_viz_goal_images,
                    goal_pos,
                    obj_inst,
                    linear_vel.cpu(),
                    angular_vel.cpu(),
                    last_poses.cpu(),
                    (0.12*select_traj).cpu(),                    
                    eval_type,
                    project_folder,
                    epoch,
                    num_images_log,
                    30,                    
                    use_wandb,
                )                
    print(eval_type, "total loss:", all_total/count_batch, "dist loss:", all_dist/count_batch, "diff loss:", all_diff/count_batch, "col loss:", all_col/count_batch, "batch count:", count_batch, "data size:", data_size)

def evaluate_nomad(
    eval_type: str,
    ema_model: EMAModel,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    noise_scheduler: DDPMScheduler,
    goal_mask_prob: float,
    project_folder: str,
    epoch: int,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    eval_fraction: float = 0.25,
    use_wandb: bool = True,
):
    """
    Evaluate the model on the given evaluation dataset.

    Args:
        eval_type (string): f"{data_type}_{eval_type}" (e.g. "recon_train", "gs_test", etc.)
        ema_model (nn.Module): exponential moving average version of model to evaluate
        dataloader (DataLoader): dataloader for eval
        transform (transforms): transform to apply to images
        device (torch.device): device to use for evaluation
        noise_scheduler: noise scheduler to evaluate with 
        project_folder (string): path to project folder
        epoch (int): current epoch
        print_log_freq (int): how often to print logs 
        wandb_log_freq (int): how often to log to wandb
        image_log_freq (int): how often to log images
        alpha (float): weight for action loss
        num_images_log (int): number of images to log
        eval_fraction (float): fraction of data to use for evaluation
        use_wandb (bool): whether to use wandb for logging
    """
    goal_mask_prob = torch.clip(torch.tensor(goal_mask_prob), 0, 1)
    ema_model = ema_model.averaged_model
    ema_model.eval()
    
    num_batches = len(dataloader)

    uc_action_loss_logger = Logger("uc_action_loss", eval_type, window_size=print_log_freq)
    uc_action_waypts_cos_sim_logger = Logger(
        "uc_action_waypts_cos_sim", eval_type, window_size=print_log_freq
    )
    uc_multi_action_waypts_cos_sim_logger = Logger(
        "uc_multi_action_waypts_cos_sim", eval_type, window_size=print_log_freq
    )
    gc_dist_loss_logger = Logger("gc_dist_loss", eval_type, window_size=print_log_freq)
    gc_action_loss_logger = Logger("gc_action_loss", eval_type, window_size=print_log_freq)
    gc_action_waypts_cos_sim_logger = Logger(
        "gc_action_waypts_cos_sim", eval_type, window_size=print_log_freq
    )
    gc_multi_action_waypts_cos_sim_logger = Logger(
        "gc_multi_action_waypts_cos_sim", eval_type, window_size=print_log_freq
    )
    loggers = {
        "uc_action_loss": uc_action_loss_logger,
        "uc_action_waypts_cos_sim": uc_action_waypts_cos_sim_logger,
        "uc_multi_action_waypts_cos_sim": uc_multi_action_waypts_cos_sim_logger,
        "gc_dist_loss": gc_dist_loss_logger,
        "gc_action_loss": gc_action_loss_logger,
        "gc_action_waypts_cos_sim": gc_action_waypts_cos_sim_logger,
        "gc_multi_action_waypts_cos_sim": gc_multi_action_waypts_cos_sim_logger,
    }
    num_batches = max(int(num_batches * eval_fraction), 1)

    with tqdm.tqdm(
        itertools.islice(dataloader, num_batches), 
        total=num_batches, 
        dynamic_ncols=True, 
        desc=f"Evaluating {eval_type} for epoch {epoch}", 
        leave=False) as tepoch:
        for i, data in enumerate(tepoch):
            (
                obs_image, 
                goal_image,
                actions,
                distance,
                goal_pos,
                dataset_idx,
                action_mask,
            ) = data
            
            obs_images = torch.split(obs_image, 3, dim=1)
            batch_viz_obs_images = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE[::-1])
            batch_viz_goal_images = TF.resize(goal_image, VISUALIZATION_IMAGE_SIZE[::-1])
            batch_obs_images = [transform(obs) for obs in obs_images]
            batch_obs_images = torch.cat(batch_obs_images, dim=1).to(device)
            batch_goal_images = transform(goal_image).to(device)
            action_mask = action_mask.to(device)

            B = actions.shape[0]

            # Generate random goal mask
            rand_goal_mask = (torch.rand((B,)) < goal_mask_prob).long().to(device)
            goal_mask = torch.ones_like(rand_goal_mask).long().to(device)
            no_mask = torch.zeros_like(rand_goal_mask).long().to(device)

            rand_mask_cond = ema_model("vision_encoder", obs_img=batch_obs_images, goal_img=batch_goal_images, input_goal_mask=rand_goal_mask)

            obsgoal_cond = ema_model("vision_encoder", obs_img=batch_obs_images, goal_img=batch_goal_images, input_goal_mask=no_mask)
            obsgoal_cond = obsgoal_cond.flatten(start_dim=1)

            goal_mask_cond = ema_model("vision_encoder", obs_img=batch_obs_images, goal_img=batch_goal_images, input_goal_mask=goal_mask)

            distance = distance.to(device)

            deltas = get_delta(actions)
            ndeltas = normalize_data(deltas, ACTION_STATS)
            naction = from_numpy(ndeltas).to(device)
            assert naction.shape[-1] == 2, "action dim must be 2"

            # Sample noise to add to actions
            noise = torch.randn(naction.shape, device=device)

            # Sample a diffusion iteration for each data point
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (B,), device=device
            ).long()

            noisy_actions = noise_scheduler.add_noise(
                naction, noise, timesteps)

            ### RANDOM MASK ERROR ###
            # Predict the noise residual
            rand_mask_noise_pred = ema_model("noise_pred_net", sample=noisy_actions, timestep=timesteps, global_cond=rand_mask_cond)
            
            # L2 loss
            rand_mask_loss = nn.functional.mse_loss(rand_mask_noise_pred, noise)
            
            ### NO MASK ERROR ###
            # Predict the noise residual
            no_mask_noise_pred = ema_model("noise_pred_net", sample=noisy_actions, timestep=timesteps, global_cond=obsgoal_cond)
            
            # L2 loss
            no_mask_loss = nn.functional.mse_loss(no_mask_noise_pred, noise)

            ### GOAL MASK ERROR ###
            # predict the noise residual
            goal_mask_noise_pred = ema_model("noise_pred_net", sample=noisy_actions, timestep=timesteps, global_cond=goal_mask_cond)
            
            # L2 loss
            goal_mask_loss = nn.functional.mse_loss(goal_mask_noise_pred, noise)
            
            # Logging
            loss_cpu = rand_mask_loss.item()
            tepoch.set_postfix(loss=loss_cpu)

            wandb.log({"diffusion_eval_loss (random masking)": rand_mask_loss})
            wandb.log({"diffusion_eval_loss (no masking)": no_mask_loss})
            wandb.log({"diffusion_eval_loss (goal masking)": goal_mask_loss})

            if i % print_log_freq == 0 and print_log_freq != 0:
                losses = _compute_losses_nomad(
                            ema_model,
                            noise_scheduler,
                            batch_obs_images,
                            batch_goal_images,
                            distance.to(device),
                            actions.to(device),
                            device,
                            action_mask.to(device),
                        )
                
                for key, value in losses.items():
                    if key in loggers:
                        logger = loggers[key]
                        logger.log_data(value.item())
            
                data_log = {}
                for key, logger in loggers.items():
                    data_log[logger.full_name()] = logger.latest()
                    if i % print_log_freq == 0 and print_log_freq != 0:
                        print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")

                if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
                    wandb.log(data_log, commit=True)

            if image_log_freq != 0 and i % image_log_freq == 0:
                visualize_diffusion_action_distribution(
                    ema_model,
                    noise_scheduler,
                    batch_obs_images,
                    batch_goal_images,
                    batch_viz_obs_images,
                    batch_viz_goal_images,
                    actions,
                    distance,
                    goal_pos,
                    device,
                    eval_type,
                    project_folder,
                    epoch,
                    num_images_log,
                    30,
                    use_wandb,
                )

###
def evaluate_exaug_dist_gnm_delay(
    eval_type: str,
    ema_model: EMAModel,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    project_folder: str,
    epoch: int,
    sacson: bool,
    model_depth,
    model_pedtraj,
    device2,         
    len_traj_pred: int,    
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    eval_fraction: float = 0.25,
    use_wandb: bool = True,    
):
    """
    Train the model for one epoch.

    Args:
        model: model to train
        ema_model: exponential moving average model
        optimizer: optimizer to use
        dataloader: dataloader for training
        transform: transform to use
        device: device to use
        noise_scheduler: noise scheduler to train with 
        project_folder: folder to save images to
        epoch: current epoch
        alpha: weight of action loss
        print_log_freq: how often to print loss
        image_log_freq: how often to log images
        num_images_log: number of images to log
        use_wandb: whether to use wandb
    """
    #goal_mask_prob = torch.clip(torch.tensor(goal_mask_prob), 0, 1)
    ema_model = ema_model  
    ema_model.eval()
    num_batches = len(dataloader)
    num_batches = max(int(num_batches * eval_fraction), 1)

    total_loss_logger = Logger("total_loss", "test", window_size=print_log_freq)
    dist_loss_logger = Logger("dist_loss", "test", window_size=print_log_freq)
    distall_loss_logger = Logger("distall_loss", "test", window_size=print_log_freq)    
    smooth_loss_logger = Logger("smooth_loss", "test", window_size=print_log_freq)
    geo_loss_logger = Logger("geo_loss", "test", window_size=print_log_freq)
    social_loss_logger = Logger("social_loss", "test", window_size=print_log_freq)
    personal_loss_logger = Logger("personal_loss", "test", window_size=print_log_freq)
    disttemp_loss_logger = Logger("disttemp_loss", "test", window_size=print_log_freq)
    goal_dist_mean_logger = Logger("goal_dist_mean", "test", window_size=print_log_freq)
    goal_dist_median_logger = Logger("goal_dist_median", "test", window_size=print_log_freq)
        
    loggers = {
        "total_loss": total_loss_logger,
        "dist_loss": dist_loss_logger,
        "distall_loss": distall_loss_logger,        
        "smooth_loss": smooth_loss_logger,
        "geo_loss": geo_loss_logger,
        "social_loss": social_loss_logger,
        "personal_loss": personal_loss_logger,     
        "disttemp_loss": disttemp_loss_logger,   
        "goal_dist_mean": goal_dist_mean_logger,   
        "goal_dist_median": goal_dist_median_logger,                    
    }
    
    mask_360 = np.loadtxt(open("./mask_360view.csv", "rb"), delimiter=",", skiprows=0)   
    mask_360_resize = np.repeat(np.expand_dims(cv2.resize(mask_360, (832, 128)), 0), 3, 0).astype(np.float32)
    mask_360_torch = torch.from_numpy(mask_360_resize[:,:,0:416]).unsqueeze(0).to(device2)

    linear_vel_old = 0.5*torch.rand(100, 8).float().to(device)
    angular_vel_old = 1.0*torch.rand(100, 8).float().to(device)
    
    with tqdm.tqdm(
        itertools.islice(dataloader, num_batches), 
        total=num_batches, 
        dynamic_ncols=True, 
        desc=f"Evaluating {eval_type} for epoch {epoch}", 
        leave=False) as tepoch:      
        for i, data in enumerate(tepoch):
            (
                obs_image,
                goal_image,
                action_label,
                dist_label,
                goal_pos,
                dataset_index,
                action_mask,
                current_image_depth,
                geoloss_range,
                local_goal_mat,
                local_yaw,
            ) = data         
            B, _, H, W = current_image_depth.size()    
            current_image_depth = current_image_depth.to(device2)                          
            obs_images = torch.split(obs_image, 3, dim=1)
                        
            viz_obs_image = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE[::-1])
            viz_obs_image_past = TF.resize(obs_images[0], VISUALIZATION_IMAGE_SIZE[::-1])   
            obs_images = [transform(obs_image).to(device) for obs_image in obs_images]
            obs_image = torch.cat(obs_images, dim=1)           
            
            with torch.no_grad():
                proj_3d, outputs = model_depth.forward(current_image_depth) #for depth360   

            batch_3d_point_cpu = proj_3d.cpu()
            batch_3d_point = batch_3d_point_cpu.to(device) 

            # Get distance label
            distance_metric = torch.sqrt(goal_pos.to(device)[:,0]**2 + goal_pos.to(device)[:,1]**2)
            fargoal_mask = ((torch.abs(local_goal_mat[:, 0,2]) < 2.0) * (torch.abs(local_goal_mat[:, 1,2]) < 2.0)).to(device)
            
            distance = dist_label.float().to(device)
            goal_mask = (distance > 0.1) * fargoal_mask
            goal_mask_zero = distance > 0.1

            for ig in range(B):
                if not goal_mask_zero[ig]:
                    distance[ig] = 20
                    igr = random.randint(0, B-1) 
                    while ig == igr:
                        igr = random.randint(0, B-1) 
                    goal_image[ig] = goal_image[igr]

            viz_goal_image = TF.resize(goal_image, VISUALIZATION_IMAGE_SIZE[::-1])
            batch_goal_pos = goal_pos.to(device)
            goal_image = transform(goal_image).to(device)

            rsize = torch.rand(B, 1, 1).to(device) #robot radius : 0 -- 1.0 m
            delay = torch.randint(0, 5, (B, 1, 1)).to(device)      

            cs = random.randint(0,2)            
            vel_past = torch.cat((linear_vel_old[:, cs:cs+6], angular_vel_old[:, cs:cs+6]), axis=1).unsqueeze(2)     
            
            with torch.no_grad():
                linear_vel, angular_vel, dist_temp = ema_model(obs_image, goal_image, rsize, delay, vel_past)

            for ig in range(B):
                linear_vel_old[ig, delay[ig,0,0]:6] *= 0.0
                angular_vel_old[ig, delay[ig,0,0]:6] *= 0.0
                                
            linear_vel_d = torch.cat((linear_vel_old, linear_vel), axis=1)
            angular_vel_d = torch.cat((angular_vel_old, angular_vel), axis=1)            
            
            linear_vel_old = linear_vel.detach()
            angular_vel_old = angular_vel.detach()                
                
            px_ref_list, pz_ref_list, ry_ref_list = robot_pos_model_fix(linear_vel_d, angular_vel_d)
            px_ref = px_ref_list[-1]
            pz_ref = pz_ref_list[-1]
            ry_ref = ry_ref_list[-1]
            last_poses = torch.cat((pz_ref.unsqueeze(1), -px_ref.unsqueeze(1)), axis=1) #from camera coordinate to robot local coordinate
       
            mat_1 = torch.cat((torch.cos(-ry_ref).unsqueeze(1), -torch.sin(-ry_ref).unsqueeze(1), 2.0*pz_ref.unsqueeze(1)), axis=1)
            mat_2 = torch.cat((torch.sin(-ry_ref).unsqueeze(1), torch.cos(-ry_ref).unsqueeze(1), -2.0*px_ref.unsqueeze(1)), axis=1)
            mat_3 = torch.cat((torch.zeros(B,1), torch.zeros(B,1), torch.ones(B,1)), axis=1).to(device)   
            last_pose_mat = torch.cat((mat_1.unsqueeze(1), mat_2.unsqueeze(1), mat_3.unsqueeze(1)), axis=1)            
            
            robot_traj_list = []
            for ip in range(len(px_ref_list)):
                mat_1 = torch.cat((torch.cos(-ry_ref_list[ip]).unsqueeze(1), -torch.sin(-ry_ref_list[ip]).unsqueeze(1), 2.0 * pz_ref_list[ip].unsqueeze(1)), axis=1)
                mat_2 = torch.cat((torch.sin(-ry_ref_list[ip]).unsqueeze(1), torch.cos(-ry_ref_list[ip]).unsqueeze(1), -2.0 * px_ref_list[ip].unsqueeze(1)), axis=1)
                mat_3 = torch.cat((torch.zeros(B,1), torch.zeros(B,1), torch.ones(B,1)), axis=1).to(device)   
                mat_combine = torch.cat((mat_1.unsqueeze(1), mat_2.unsqueeze(1), mat_3.unsqueeze(1)), axis=1)
                robot_traj_list.append(mat_combine.unsqueeze(1))
            robot_traj_vec = torch.cat(robot_traj_list, axis=1)

            local_goal_mat[:, 0,2] *= 2.0                    
            local_goal_mat[:, 1,2] *= 2.0  
            local_goal_vec = local_goal_mat.unsqueeze(1).repeat(1,8,1,1)

            dist_loss = nn.functional.mse_loss(last_pose_mat[goal_mask], local_goal_mat.to(device)[goal_mask])                                    
            distall_loss = nn.functional.mse_loss(robot_traj_vec[goal_mask, 6:14], local_goal_vec.to(device)[goal_mask])                     
            diff_loss = nn.functional.mse_loss(linear_vel[:,:-1][goal_mask], linear_vel[:,1:][goal_mask]) + nn.functional.mse_loss(angular_vel[:,:-1][goal_mask], angular_vel[:,1:][goal_mask]) 

            # Get distance label
            distance = distance.float().to(device)

            # Predict distance         
            dist_temp_loss = F.mse_loss(dist_temp.squeeze(-1), distance)

            PC3D = []
            for j in range(len_traj_pred):
                px_ref = px_ref_list[j+6]
                pz_ref = pz_ref_list[j+6]
                ry_ref = ry_ref_list[j+6]                

                Tod = torch.zeros((B, 4, 4)).to(device)
                Tod[:, 0, 0] = torch.cos(ry_ref)
                Tod[:, 0, 2] = torch.sin(ry_ref)
                Tod[:, 1, 1] = 1.0
                Tod[:, 2, 0] = -torch.sin(ry_ref)
                Tod[:, 2, 2] = torch.cos(ry_ref)
                Tod[:, 0, 3] = px_ref
                Tod[:, 2, 3] = pz_ref
                Tod[:, 3, 3] = 1.0

                Ttrans = torch.inverse(Tod)[:, :3, :]               
                batch_3d_point_x = torch.cat((batch_3d_point.view(B, 3, -1), torch.ones(B,1,416*128).to(device)), axis=1)
                cam_points_trans = torch.matmul(Ttrans, batch_3d_point_x).view(B, 3, 128, 416)
                PC3D.append(cam_points_trans.unsqueeze(1))                                                  
            
            PC3D_cat = torch.cat(PC3D, axis=1)                    
  
            loss_geo = geometry_criterion_range(PC3D_cat[goal_mask], rsize[:,:,0][goal_mask], len_traj_pred, geoloss_range.to(device)[goal_mask], device)
            norm = 10.0
            
            if sacson:
                ped_past = torch.cat((-torch.flip(ped_list[:,:,1], dims=[1]), torch.flip(ped_list[:,:,0], dims=[1])), axis=1).to(device)  
                robot_past = torch.cat((-torch.flip(robot_list[:,:,1], dims=[1]), torch.flip(robot_list[:,:,0], dims=[1])), axis=1).to(device)
                
                robot_future_x = []
                robot_future_z = []
                for ir in range(len(px_ref_list)):
                    robot_future_x.append(px_ref_list[ir].unsqueeze(1))
                    robot_future_z.append(pz_ref_list[ir].unsqueeze(1))                    
                robot_future = torch.cat((torch.cat(robot_future_x, axis=1), torch.cat(robot_future_z, axis=1)), axis=1).to(device)  
                            
                ped_past_c = torch.clamp(ped_past, min=-10.0, max=10.0)/norm
                robot_past_c = torch.clamp(robot_past, min=-10.0, max=10.0)/norm
                robot_future_c = torch.clamp(robot_future, min=-10.0, max=10.0)/norm
                
                flag_ped = (ped_past_c[:,0] != 0.0) & (ped_past_c[:,8] != 0.0)
                
                robot_past_c = 0.333*robot_past_c #frodbot is much faster than Vizbot.
                delta_est_ped_traj = model_pedtraj(ped_past_c, robot_past_c, robot_future_c)
                delta_est_ped_traj_zero = model_pedtraj(ped_past_c, robot_past_c, 0.0*robot_future_c)

                traj_x = torch.cumsum(delta_est_ped_traj[:,0:8]/norm, dim=1) + ped_past_c[:,0:1].repeat(1,8)
                traj_y = torch.cumsum(delta_est_ped_traj[:,8:16]/norm, dim=1) + ped_past_c[:,8:9].repeat(1,8)           
                est_ped_traj = torch.clamp(torch.cat((traj_x, traj_y), axis=1), min=-10.0, max=10.0)

                delta_est_ped_traj_zero_detach = delta_est_ped_traj_zero.detach()
                traj_xz = torch.cumsum(delta_est_ped_traj_zero_detach[:,0:8]/norm, dim=1) + ped_past_c[:,0:1].repeat(1,8)
                traj_yz = torch.cumsum(delta_est_ped_traj_zero_detach[:,8:16]/norm, dim=1) + ped_past_c[:,8:9].repeat(1,8)           
                est_ped_traj_zeros = torch.clamp(torch.cat((traj_xz, traj_yz), axis=1), min=-10.0, max=10.0)
                
                social_loss = nn.functional.mse_loss(est_ped_traj[flag_ped*goal_mask], est_ped_traj_zeros[flag_ped*goal_mask])
                
                max_pl = rsize.squeeze(1).repeat(1,8) + 0.5
                min_pl = rsize.squeeze(1).repeat(1,8) * 0.0
                personal_loss = nn.functional.mse_loss(max_pl[flag_ped*goal_mask], torch.clamp(torch.sqrt((norm*robot_future_c[flag_ped*goal_mask][:,0:8] - norm*est_ped_traj[flag_ped*goal_mask][:,0:8])**2 + (norm*robot_future_c[flag_ped*goal_mask][:,8:16] - norm*est_ped_traj[flag_ped*goal_mask][:,8:16])**2), min=min_pl[flag_ped*goal_mask], max= max_pl[flag_ped*goal_mask]))
            else:
                est_ped_traj = torch.zeros(B, 16)
                est_ped_traj_zeros = torch.zeros(B, 16)      
                ped_past_c = torch.zeros(B, 16)
                robot_past_c = torch.zeros(B, 16)
                social_loss = nn.functional.mse_loss(est_ped_traj, est_ped_traj_zeros)
                personal_loss = nn.functional.mse_loss(est_ped_traj, est_ped_traj_zeros)
            
            loss = 4.0*dist_loss + 0.4*distall_loss + 0.5*diff_loss + 10.0*loss_geo + 100.0*social_loss + 10.0*personal_loss + 0.001*dist_temp_loss 
            goal_dist_mean = torch.max(torch.sqrt(goal_pos.to(device)[:,0][goal_mask]**2 + goal_pos.to(device)[:,1][goal_mask]**2))
            goal_dist_median = torch.min(torch.sqrt(goal_pos.to(device)[:,0][goal_mask]**2 + goal_pos.to(device)[:,1][goal_mask]**2))
            # Logging            
            loss_cpu = loss.cpu().item()          
            
            tepoch.set_postfix(loss=loss_cpu)
            wandb.log({"total_loss": loss_cpu})
            wandb.log({"dist_loss": dist_loss.item()})
            wandb.log({"distall_loss": distall_loss.item()})            
            wandb.log({"smooth_loss": diff_loss.item()})
            wandb.log({"geo_loss": loss_geo.item()})
            wandb.log({"social_loss": social_loss.item()})
            wandb.log({"personal_loss": personal_loss.item()})
            wandb.log({"disttemp_loss": dist_temp_loss.item()})
            wandb.log({"goal_dist_mean": goal_dist_mean.item()})
            wandb.log({"goal_dist_median": goal_dist_median.item()})
                                    
            if i % print_log_freq == 0:
                losses = {}
                losses['total_loss'] = loss_cpu
                losses['dist_loss'] = dist_loss.item()
                losses['distall_loss'] = distall_loss.item()                
                losses['smooth_loss'] = diff_loss.item()                 
                losses['geo_loss'] = loss_geo.item()
                losses['social_loss'] = social_loss.item()                 
                losses['personal_loss'] = personal_loss.item()  
                losses['disttemp_loss'] = dist_temp_loss.item()  
                losses['goal_dist_mean'] = goal_dist_mean.item()  
                losses['goal_dist_median'] = goal_dist_median.item()  
                                                                
                for key, value in losses.items():
                    if key in loggers:
                        logger = loggers[key]
                        logger.log_data(value)
            
                data_log = {}
                for key, logger in loggers.items():
                    data_log[logger.full_name()] = logger.latest()
                    if i % print_log_freq == 0 and print_log_freq != 0:
                        print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")

                if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
                    wandb.log(data_log, commit=True)

            if image_log_freq != 0 and i % image_log_freq == 0:
                visualize_exaug_delay_estimation(
                    viz_obs_image, 
                    viz_obs_image_past,                     
                    viz_goal_image,
                    batch_3d_point,
                    goal_pos,
                    local_yaw,
                    linear_vel_d.cpu(),
                    angular_vel_d.cpu(),
                    norm*ped_past_c.cpu(),
                    norm*est_ped_traj.cpu(),
                    norm*est_ped_traj_zeros.cpu(),
                    norm*robot_past_c.cpu(),
                    last_poses.cpu(),
                    rsize.cpu(),
                    "test",
                    project_folder,
                    epoch,
                    num_images_log,
                    30,                    
                    use_wandb,
                    )    
###
def evaluate_il_dist_gnm(
    eval_type: str,
    ema_model: EMAModel,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    project_folder: str,
    epoch: int,
    sacson: bool,
    #model_depth,
    #model_pedtraj,
    #device2,         
    len_traj_pred: int,    
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    eval_fraction: float = 0.25,
    use_wandb: bool = True,    
):
    """
    Train the model for one epoch.

    Args:
        model: model to train
        ema_model: exponential moving average model
        optimizer: optimizer to use
        dataloader: dataloader for training
        transform: transform to use
        device: device to use
        noise_scheduler: noise scheduler to train with 
        project_folder: folder to save images to
        epoch: current epoch
        alpha: weight of action loss
        print_log_freq: how often to print loss
        image_log_freq: how often to log images
        num_images_log: number of images to log
        use_wandb: whether to use wandb
    """
    #goal_mask_prob = torch.clip(torch.tensor(goal_mask_prob), 0, 1)
    ema_model = ema_model  
    ema_model.eval()
    num_batches = len(dataloader)
    num_batches = max(int(num_batches * eval_fraction), 1)

    total_loss_logger = Logger("total_loss", "test", window_size=print_log_freq)
    dist_loss_logger = Logger("dist_loss", "test", window_size=print_log_freq)
    action_loss_logger = Logger("action_loss", "test", window_size=print_log_freq)    
    
    loggers = {
        "total_loss": total_loss_logger,
        "dist_loss": dist_loss_logger,
        "action_loss": action_loss_logger,                     
    }
    with tqdm.tqdm(
        itertools.islice(dataloader, num_batches), 
        total=num_batches, 
        dynamic_ncols=True, 
        desc=f"Evaluating {eval_type} for epoch {epoch}", 
        leave=False) as tepoch:      
        for i, data in enumerate(tepoch):
            (
                    obs_image,
                    goal_image,
                    actions,
                    distance,
                    goal_pos,
                    dataset_index,
                    action_mask,
                    #_,
                    #_,
                ) = data                   
            #at different GPUs
            if psutil.virtual_memory().percent > 90.0:
                print("RAM usage (%)", psutil.virtual_memory().percent)
                break                    
            B, _, H, W = obs_image.size()  
            local_yaw = torch.ones(B)
              
            obs_images = torch.split(obs_image, 3, dim=1)
            viz_obs_image = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE[::-1])
            viz_obs_image_past = TF.resize(obs_images[0], VISUALIZATION_IMAGE_SIZE[::-1])   
            obs_images = [transform(obs_image).to(device) for obs_image in obs_images]
            obs_image = torch.cat(obs_images, dim=1)
            actions = actions.to(device)
            
            action_mask = action_mask.to(device)
            batch_goal_pos = goal_pos.to(device)
            far_goal_mask = (torch.abs(batch_goal_pos[:,0]) < 5.0) * (torch.abs(batch_goal_pos[:,1]) < 5.0)
            distance = distance.float().to(device)
            goal_mask = (distance > 0.1) * far_goal_mask

            for ig in range(B):
                if not goal_mask[ig]:
                    distance[ig] = 20
                    igr = random.randint(0, B-1) 
                    while ig == igr:
                        igr = random.randint(0, B-1) 
                    goal_image[ig] = goal_image[igr]

            viz_goal_image = TF.resize(goal_image, VISUALIZATION_IMAGE_SIZE[::-1])
            batch_goal_pos = goal_pos.to(device)
            goal_image = transform(goal_image).to(device)      
            
            with torch.no_grad():
                dist_pred, action_pred = ema_model(obs_image, goal_image)
            
            losses = _compute_losses(
                dist_label=distance,
                action_label=actions,
                dist_pred=dist_pred,
                action_pred=action_pred,
                alpha=0.5,
                learn_angle=True,
                action_mask=action_mask,
            )
            
            # Logging            
            loss_cpu = losses["total_loss"].item()
            dist_loss_cpu = losses["dist_loss"].item()
            action_loss_cpu = losses["action_loss"].item()          
            
            tepoch.set_postfix(loss=loss_cpu)
            wandb.log({"total_loss": loss_cpu})
            wandb.log({"dist_loss": dist_loss_cpu})
            wandb.log({"action_loss": action_loss_cpu})                  
                        
            if i % print_log_freq == 0:
                losses = {}
                losses['total_loss'] = loss_cpu
                losses['dist_loss'] = dist_loss_cpu
                losses['action_loss'] = action_loss_cpu   
                                
                for key, value in losses.items():
                    if key in loggers:
                        logger = loggers[key]
                        logger.log_data(value)
            
                data_log = {}
                for key, logger in loggers.items():
                    data_log[logger.full_name()] = logger.latest()
                    if i % print_log_freq == 0 and print_log_freq != 0:
                        print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")

                if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
                    wandb.log(data_log, commit=True)

            if image_log_freq != 0 and i % image_log_freq == 0:
                visualize_il_estimation(
                    viz_obs_image, 
                    viz_obs_image_past,                     
                    viz_goal_image,
                    goal_pos,
                    local_yaw,
                    action_pred,
                    "test",
                    project_folder,
                    epoch,
                    num_images_log,
                    30,                    
                    use_wandb,
                    )                  
                  
###
def evaluate_il_dist_gnm_gps(
    eval_type: str,
    ema_model: EMAModel,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    project_folder: str,
    epoch: int,
    sacson: bool,   
    len_traj_pred: int,    
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    eval_fraction: float = 0.25,
    use_wandb: bool = True,    
):
    """
    Train the model for one epoch.

    Args:
        model: model to train
        ema_model: exponential moving average model
        optimizer: optimizer to use
        dataloader: dataloader for training
        transform: transform to use
        device: device to use
        noise_scheduler: noise scheduler to train with 
        project_folder: folder to save images to
        epoch: current epoch
        alpha: weight of action loss
        print_log_freq: how often to print loss
        image_log_freq: how often to log images
        num_images_log: number of images to log
        use_wandb: whether to use wandb
    """
    ema_model = ema_model  
    ema_model.eval()
    num_batches = len(dataloader)
    num_batches = max(int(num_batches * eval_fraction), 1)

    total_loss_logger = Logger("total_loss", "test", window_size=print_log_freq)
    action_loss_logger = Logger("action_loss", "test", window_size=print_log_freq)    
    
    loggers = {
        "total_loss": total_loss_logger,
        "action_loss": action_loss_logger,                     
    }

    with tqdm.tqdm(
        itertools.islice(dataloader, num_batches), 
        total=num_batches, 
        dynamic_ncols=True, 
        desc=f"Evaluating {eval_type} for epoch {epoch}", 
        leave=False) as tepoch:      
        for i, data in enumerate(tepoch):
            (
                    obs_image,
                    goal_image,
                    actions,
                    distance,
                    goal_pos,
                    goal_yaw,
                    dataset_index,
                    action_mask,
                    #_,
                    #_,
                ) = data                   
            #at different GPUs
            if psutil.virtual_memory().percent > 90.0:
                print("RAM usage (%)", psutil.virtual_memory().percent)
                break      
            B, _, H, W = obs_image.size()  
            local_yaw = torch.ones(B)
              
            obs_images = torch.split(obs_image, 3, dim=1)
            viz_obs_image = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE[::-1])
            viz_obs_image_past = TF.resize(obs_images[0], VISUALIZATION_IMAGE_SIZE[::-1])   
            obs_images = [transform(obs_image).to(device) for obs_image in obs_images]
            obs_image = torch.cat(obs_images, dim=1)
            actions = actions.to(device)
            action_mask = action_mask.to(device)
            
            batch_goal_pos = goal_pos.to(device)
            far_goal_mask = (torch.abs(batch_goal_pos[:,0]) < 5.0) * (torch.abs(batch_goal_pos[:,1]) < 5.0)

            distance = distance.float().to(device)
            goal_mask = (distance > 0.1) * far_goal_mask
            goal_pose_gps = torch.cat((goal_pos, torch.cos(goal_yaw).unsqueeze(1), torch.sin(goal_yaw).unsqueeze(1)), axis=1).to(device)

            for ig in range(B):
                if not goal_mask[ig]:
                    distance[ig] = 20
                    igr = random.randint(0, B-1) 
                    while ig == igr:
                        igr = random.randint(0, B-1) 
                    goal_image[ig] = goal_image[igr]

            viz_goal_image = TF.resize(goal_image, VISUALIZATION_IMAGE_SIZE[::-1])
            batch_goal_pos = goal_pos.to(device)
            goal_image = transform(goal_image).to(device)   
            
            with torch.no_grad():
                action_pred = ema_model(obs_image, goal_pose_gps)

            losses = _compute_losses_gps(
                action_label=actions,
                action_pred=action_pred,
                learn_angle=True,
                action_mask=action_mask,
            )
            
            # Logging            
            loss_cpu = losses["total_loss"].item()
            action_loss_cpu = losses["action_loss"].item()          
            
            tepoch.set_postfix(loss=loss_cpu)
            wandb.log({"total_loss": loss_cpu})
            wandb.log({"action_loss": action_loss_cpu})                  
                        
            if i % print_log_freq == 0:
                losses = {}
                losses['total_loss'] = loss_cpu
                losses['action_loss'] = action_loss_cpu   
                                
                for key, value in losses.items():
                    if key in loggers:
                        logger = loggers[key]
                        logger.log_data(value)
            
                data_log = {}
                for key, logger in loggers.items():
                    data_log[logger.full_name()] = logger.latest()
                    if i % print_log_freq == 0 and print_log_freq != 0:
                        print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")

                if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
                    wandb.log(data_log, commit=True)

            if image_log_freq != 0 and i % image_log_freq == 0:
                visualize_il_estimation_gps(
                    viz_obs_image, 
                    viz_obs_image_past,                     
                    viz_goal_image,
                    goal_pos,
                    local_yaw,
                    action_pred,
                    actions,
                    "test",
                    project_folder,
                    epoch,
                    num_images_log,
                    30,                    
                    use_wandb,
                    )   
###
def evaluate_il_dist_gnm_gps_map(
    eval_type: str,
    ema_model: EMAModel,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    project_folder: str,
    epoch: int,
    sacson: bool, 
    len_traj_pred: int,    
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    eval_fraction: float = 0.25,
    use_wandb: bool = True,    
):
    """
    Train the model for one epoch.

    Args:
        model: model to train
        ema_model: exponential moving average model
        optimizer: optimizer to use
        dataloader: dataloader for training
        transform: transform to use
        device: device to use
        noise_scheduler: noise scheduler to train with 
        project_folder: folder to save images to
        epoch: current epoch
        alpha: weight of action loss
        print_log_freq: how often to print loss
        image_log_freq: how often to log images
        num_images_log: number of images to log
        use_wandb: whether to use wandb
    """
    ema_model = ema_model  
    ema_model.eval()
    num_batches = len(dataloader)
    num_batches = max(int(num_batches * eval_fraction), 1)

    total_loss_logger = Logger("total_loss", "test", window_size=print_log_freq)
    action_loss_logger = Logger("action_loss", "test", window_size=print_log_freq)    
    
    loggers = {
        "total_loss": total_loss_logger,
        "action_loss": action_loss_logger,                     
    }
    
    D = np.array(np.load('/media/noriaki/Noriaki_Data/learning-language-navigation_SACSoN/train/vint_train/training/fisheye_calibration/D.npy', mmap_mode='r'))
    K = np.array(np.load('/media/noriaki/Noriaki_Data/learning-language-navigation_SACSoN/train/vint_train/training/fisheye_calibration/K.npy', mmap_mode='r'))

    mask_360 = np.loadtxt(open("/media/noriaki/Noriaki_Data/learning-language-navigation_SACSoN/train/mask_360view.csv", "rb"), delimiter=",", skiprows=0)    
    mask_360_resize = np.repeat(np.expand_dims(cv2.resize(mask_360, (832, 128)), 0), 3, 0).astype(np.float32)
    mask_360_torch = torch.from_numpy(mask_360_resize[:,:,0:416]).unsqueeze(0).to(device2)
    
    with tqdm.tqdm(
        itertools.islice(dataloader, num_batches), 
        total=num_batches, 
        dynamic_ncols=True, 
        desc=f"Evaluating {eval_type} for epoch {epoch}", 
        leave=False) as tepoch:      
        for i, data in enumerate(tepoch): 
            (
                    obs_image,
                    goal_image,
                    actions,
                    distance,
                    goal_pos,
                    goal_yaw,
                    dataset_index,
                    action_mask,
                    #_,
                    #_,
                ) = data                   
            #at different GPUs
            if psutil.virtual_memory().percent > 90.0:
                print("RAM usage (%)", psutil.virtual_memory().percent)
                break                   
            B, _, H, W = obs_image.size()  
            current_map_image = torch.zeros(B, 3, 96, 96)
            goal_map_image = torch.zeros(B, 3, 96, 96)            
            local_yaw = torch.ones(B)
              
            obs_images = torch.split(obs_image, 3, dim=1)
            viz_obs_image = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE[::-1])
            viz_obs_image_past = TF.resize(obs_images[0], VISUALIZATION_IMAGE_SIZE[::-1])   
            obs_images = [transform(obs_image).to(device) for obs_image in obs_images]
            obs_image_map = obs_images[0].to(device)        
            obs_image = torch.cat(obs_images, dim=1)
            
            map_images = torch.cat((current_map_image.to(device), goal_map_image.to(device), obs_image_map), axis=1)                
            actions = actions.to(device)           
            action_mask = action_mask.to(device)           
            batch_goal_pos = goal_pos.to(device)
            far_goal_mask = (torch.abs(batch_goal_pos[:,0]) < 5.0) * (torch.abs(batch_goal_pos[:,1]) < 5.0)

            distance = distance.float().to(device)
            goal_mask = (distance > 0.1) * far_goal_mask
            goal_pose_gps = torch.cat((goal_pos, torch.cos(goal_yaw).unsqueeze(1), torch.sin(goal_yaw).unsqueeze(1)), axis=1).to(device)

            for ig in range(B):
                if not goal_mask[ig]:
                    distance[ig] = 20
                    igr = random.randint(0, B-1) 
                    while ig == igr:
                        igr = random.randint(0, B-1) 
                    goal_image[ig] = goal_image[igr]

            viz_goal_image = TF.resize(goal_image, VISUALIZATION_IMAGE_SIZE[::-1])
            batch_goal_pos = goal_pos.to(device)
            goal_image = transform(goal_image).to(device)

            #model calculate            
            goal_mask_sub = torch.zeros(B).to(device) #GNM: Goal pose only            
            with torch.no_grad():
                action_pred, _ = ema_model(obs_image, goal_pose_gps, map_images, goal_mask_sub)
            
            losses = _compute_losses_gps(
                action_label=actions,
                action_pred=action_pred,
                learn_angle=True,
                action_mask=action_mask,
            )
            
            # Logging            
            loss_cpu = losses["total_loss"].item()
            action_loss_cpu = losses["action_loss"].item()           
            
            tepoch.set_postfix(loss=loss_cpu)
            wandb.log({"total_loss": loss_cpu})
            wandb.log({"action_loss": action_loss_cpu})                  
                        
            if i % print_log_freq == 0:
                losses = {}
                losses['total_loss'] = loss_cpu
                losses['action_loss'] = action_loss_cpu   
                                
                for key, value in losses.items():
                    if key in loggers:
                        logger = loggers[key]
                        logger.log_data(value)
            
                data_log = {}
                for key, logger in loggers.items():
                    data_log[logger.full_name()] = logger.latest()
                    if i % print_log_freq == 0 and print_log_freq != 0:
                        print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")

                if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
                    wandb.log(data_log, commit=True)

            if image_log_freq != 0 and i % image_log_freq == 0:
                visualize_il_estimation_gps(
                    viz_obs_image, 
                    viz_obs_image_past,                     
                    viz_goal_image,
                    goal_pos,
                    local_yaw,
                    action_pred,
                    actions,
                    "test",
                    project_folder,
                    epoch,
                    num_images_log,
                    30,                    
                    use_wandb,
                    )

# normalize data
def get_data_stats(data):
    data = data.reshape(-1,data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return stats

def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data

def get_delta(actions):
    # append zeros to first action
    ex_actions = np.concatenate([np.zeros((actions.shape[0],1,actions.shape[-1])), actions], axis=1)
    delta = ex_actions[:,1:] - ex_actions[:,:-1]
    return delta

def get_action(diffusion_output, action_stats=ACTION_STATS):
    # diffusion_output: (B, 2*T+1, 1)
    # return: (B, T-1)
    device = diffusion_output.device
    ndeltas = diffusion_output
    ndeltas = ndeltas.reshape(ndeltas.shape[0], -1, 2)
    ndeltas = to_numpy(ndeltas)
    ndeltas = unnormalize_data(ndeltas, action_stats)
    actions = np.cumsum(ndeltas, axis=1)
    return from_numpy(actions).to(device)


def model_output(
    model: nn.Module,
    noise_scheduler: DDPMScheduler,
    batch_obs_images: torch.Tensor,
    batch_goal_images: torch.Tensor,
    pred_horizon: int,
    action_dim: int,
    num_samples: int,
    device: torch.device,
):
    goal_mask = torch.ones((batch_goal_images.shape[0],)).long().to(device)
    obs_cond = model("vision_encoder", obs_img=batch_obs_images, goal_img=batch_goal_images, input_goal_mask=goal_mask)
    obs_cond = obs_cond.repeat_interleave(num_samples, dim=0)

    no_mask = torch.zeros((batch_goal_images.shape[0],)).long().to(device)
    obsgoal_cond = model("vision_encoder", obs_img=batch_obs_images, goal_img=batch_goal_images, input_goal_mask=no_mask)
    obsgoal_cond = obsgoal_cond.repeat_interleave(num_samples, dim=0)

    # initialize action from Gaussian noise
    noisy_diffusion_output = torch.randn(
        (len(obs_cond), pred_horizon, action_dim), device=device)
    diffusion_output = noisy_diffusion_output


    for k in noise_scheduler.timesteps[:]:
        # predict noise
        noise_pred = model(
            "noise_pred_net",
            sample=diffusion_output,
            timestep=k.unsqueeze(-1).repeat(diffusion_output.shape[0]).to(device),
            global_cond=obs_cond
        )

        # inverse diffusion step (remove noise)
        diffusion_output = noise_scheduler.step(
            model_output=noise_pred,
            timestep=k,
            sample=diffusion_output
        ).prev_sample

    uc_actions = get_action(diffusion_output, ACTION_STATS)

    # initialize action from Gaussian noise
    noisy_diffusion_output = torch.randn(
        (len(obs_cond), pred_horizon, action_dim), device=device)
    diffusion_output = noisy_diffusion_output

    for k in noise_scheduler.timesteps[:]:
        # predict noise
        noise_pred = model(
            "noise_pred_net",
            sample=diffusion_output,
            timestep=k.unsqueeze(-1).repeat(diffusion_output.shape[0]).to(device),
            global_cond=obsgoal_cond
        )

        # inverse diffusion step (remove noise)
        diffusion_output = noise_scheduler.step(
            model_output=noise_pred,
            timestep=k,
            sample=diffusion_output
        ).prev_sample
    obsgoal_cond = obsgoal_cond.flatten(start_dim=1)
    gc_actions = get_action(diffusion_output, ACTION_STATS)
    gc_distance = model("dist_pred_net", obsgoal_cond=obsgoal_cond)

    return {
        'uc_actions': uc_actions,
        'gc_actions': gc_actions,
        'gc_distance': gc_distance,
    }

def supervision_from_nomad(
    ema_model: nn.Module,
    noise_scheduler: DDPMScheduler,
    batch_obs_images: torch.Tensor,
    batch_goal_images: torch.Tensor,
    batch_viz_obs_images: torch.Tensor,
    batch_viz_goal_images: torch.Tensor,
    batch_goal_pos: torch.Tensor,
    device: torch.device,
    project_folder: str,
    epoch: int,
    num_images_log: int,
    it_num: int,    
    num_samples: int = 30,
    use_wandb: bool = True,
):
    """Plot samples from the exploration model."""

    max_batch_size = batch_obs_images.shape[0]

    num_images_log = min(num_images_log, batch_obs_images.shape[0], batch_goal_images.shape[0], batch_goal_pos.shape[0])
    batch_obs_images = batch_obs_images[:num_images_log]
    batch_goal_images = batch_goal_images[:num_images_log]
    
    #wandb_list = []
    pred_horizon = 8
    action_dim = 2
    
    # split into batches
    batch_obs_images_list = torch.split(batch_obs_images, max_batch_size, dim=0)
    batch_goal_images_list = torch.split(batch_goal_images, max_batch_size, dim=0)

    gc_actions_torch_list = []    
    gc_actions_list = []

    for obs, goal in zip(batch_obs_images_list, batch_goal_images_list):
        model_output_dict = model_output(
            ema_model,
            noise_scheduler,
            obs,
            goal,
            pred_horizon,
            action_dim,
            num_samples,
            device,
        )
        gc_actions_torch_list.append(model_output_dict['gc_actions'])        
    gc_actions_torch_list = torch.concat(gc_actions_torch_list, axis=0)    
    gc_actions_torch_list = torch.split(gc_actions_torch_list, num_samples, dim=0)    
    
    select_traj_list = []
    for i in range(num_images_log):
        gc_actions_torch = gc_actions_torch_list[i]
        gc_actions_torch_cat = torch.concat(torch.split(gc_actions_torch, 1, dim=1), axis=0).squeeze(1)  
        
        batch_goal_pos_i = torch.tensor([batch_goal_pos[i][1], -batch_goal_pos[i][0]])    
        device = gc_actions_torch_cat.get_device()
        
        batch_goal_pos_repeat = batch_goal_pos_i.unsqueeze(0).repeat(num_samples*8, 1).to(device)
        traj_id_all = torch.argmin(torch.sum((batch_goal_pos_repeat - gc_actions_torch_cat)**2, axis=1))
        traj_id = traj_id_all % num_samples
        select_traj_list.append(gc_actions_torch[traj_id:traj_id+1])
    return torch.concat(select_traj_list, axis=0)

def visualize_diffusion_action_distribution(
    ema_model: nn.Module,
    noise_scheduler: DDPMScheduler,
    batch_obs_images: torch.Tensor,
    batch_goal_images: torch.Tensor,
    batch_viz_obs_images: torch.Tensor,
    batch_viz_goal_images: torch.Tensor,
    batch_action_label: torch.Tensor,
    batch_distance_labels: torch.Tensor,
    batch_goal_pos: torch.Tensor,
    device: torch.device,
    eval_type: str,
    project_folder: str,
    epoch: int,
    num_images_log: int,
    num_samples: int = 30,
    use_wandb: bool = True,
):
    """Plot samples from the exploration model."""

    visualize_path = os.path.join(
        project_folder,
        "visualize",
        eval_type,
        f"epoch{epoch}",
        "action_sampling_prediction",
    )
    if not os.path.isdir(visualize_path):
        os.makedirs(visualize_path)

    max_batch_size = batch_obs_images.shape[0]

    num_images_log = min(num_images_log, batch_obs_images.shape[0], batch_goal_images.shape[0], batch_action_label.shape[0], batch_goal_pos.shape[0])
    batch_obs_images = batch_obs_images[:num_images_log]
    batch_goal_images = batch_goal_images[:num_images_log]
    batch_action_label = batch_action_label[:num_images_log]
    batch_goal_pos = batch_goal_pos[:num_images_log]
    
    wandb_list = []

    pred_horizon = batch_action_label.shape[1]
    action_dim = batch_action_label.shape[2]

    # split into batches
    batch_obs_images_list = torch.split(batch_obs_images, max_batch_size, dim=0)
    batch_goal_images_list = torch.split(batch_goal_images, max_batch_size, dim=0)

    uc_actions_list = []
    gc_actions_list = []
    gc_distances_list = []

    for obs, goal in zip(batch_obs_images_list, batch_goal_images_list):
        model_output_dict = model_output(
            ema_model,
            noise_scheduler,
            obs,
            goal,
            pred_horizon,
            action_dim,
            num_samples,
            device,
        )
        uc_actions_list.append(to_numpy(model_output_dict['uc_actions']))
        gc_actions_list.append(to_numpy(model_output_dict['gc_actions']))
        gc_distances_list.append(to_numpy(model_output_dict['gc_distance']))

    # concatenate
    uc_actions_list = np.concatenate(uc_actions_list, axis=0)
    gc_actions_list = np.concatenate(gc_actions_list, axis=0)
    gc_distances_list = np.concatenate(gc_distances_list, axis=0)

    # split into actions per observation
    uc_actions_list = np.split(uc_actions_list, num_images_log, axis=0)
    gc_actions_list = np.split(gc_actions_list, num_images_log, axis=0)
    gc_distances_list = np.split(gc_distances_list, num_images_log, axis=0)

    gc_distances_avg = [np.mean(dist) for dist in gc_distances_list]
    gc_distances_std = [np.std(dist) for dist in gc_distances_list]

    assert len(uc_actions_list) == len(gc_actions_list) == num_images_log

    np_distance_labels = to_numpy(batch_distance_labels)

    for i in range(num_images_log):
        fig, ax = plt.subplots(1, 3)
        uc_actions = uc_actions_list[i]
        gc_actions = gc_actions_list[i]
        action_label = to_numpy(batch_action_label[i])

        traj_list = np.concatenate([
            uc_actions,
            gc_actions,
            action_label[None],
        ], axis=0)
        # traj_labels = ["r", "GC", "GC_mean", "GT"]
        traj_colors = ["red"] * len(uc_actions) + ["green"] * len(gc_actions) + ["magenta"]
        traj_alphas = [0.1] * (len(uc_actions) + len(gc_actions)) + [1.0]

        # make points numpy array of robot positions (0, 0) and goal positions
        point_list = [np.array([0, 0]), to_numpy(batch_goal_pos[i])]
        point_colors = ["green", "red"]
        point_alphas = [1.0, 1.0]

        plot_trajs_and_points(
            ax[0],
            traj_list,
            point_list,
            traj_colors,
            point_colors,
            traj_labels=None,
            point_labels=None,
            quiver_freq=0,
            traj_alphas=traj_alphas,
            point_alphas=point_alphas, 
        )
        
        obs_image = to_numpy(batch_viz_obs_images[i])
        goal_image = to_numpy(batch_viz_goal_images[i])
        # move channel to last dimension
        obs_image = np.moveaxis(obs_image, 0, -1)
        goal_image = np.moveaxis(goal_image, 0, -1)
        ax[1].imshow(obs_image)
        ax[2].imshow(goal_image)

        # set title
        ax[0].set_title(f"diffusion action predictions")
        ax[1].set_title(f"observation")
        ax[2].set_title(f"goal: label={np_distance_labels[i]} gc_dist={gc_distances_avg[i]:.2f}±{gc_distances_std[i]:.2f}")
        
        # make the plot large
        fig.set_size_inches(18.5, 10.5)

        save_path = os.path.join(visualize_path, f"sample_{i}.png")
        plt.savefig(save_path)
        wandb_list.append(wandb.Image(save_path))
        plt.close(fig)
    if len(wandb_list) > 0 and use_wandb:
        wandb.log({f"{eval_type}_action_samples": wandb_list}, commit=False)

def visualize_exaug_delay_estimation(
    batch_viz_obs_images: torch.Tensor,
    batch_viz_obs_images_past: torch.Tensor,    
    batch_viz_goal_images: torch.Tensor,
    batch_3d_point: torch.Tensor,
    obj_poses: torch.Tensor,
    local_yaw: torch.Tensor,
    linear_vel: torch.Tensor,
    angular_vel: torch.Tensor,
    ped_list: torch.Tensor, 
    est_ped_traj: torch.Tensor,
    est_ped_traj_zeros: torch.Tensor,   
    robot_list: torch.Tensor,
    last_poses: torch.Tensor,
    rsize: torch.Tensor,
    eval_type: str,    
    project_folder: str,
    epoch: int,
    num_images_log: int,
    num_samples: int = 30,    
    use_wandb: bool = True,
):
    """Plot samples from the exploration model."""

    visualize_path = os.path.join(
        project_folder,
        "visualize",
        eval_type,
        f"epoch{epoch}",
        "action_sampling_prediction",
    )
    if not os.path.isdir(visualize_path):
        os.makedirs(visualize_path)

    num_images_log = min(num_images_log, batch_viz_obs_images.shape[0], batch_viz_goal_images.shape[0], obj_poses.shape[0], last_poses.shape[0])    
    batch_linear_vel = linear_vel[:num_images_log]
    batch_angular_vel = angular_vel[:num_images_log]
    
    px_list, pz_list, ry_list = robot_pos_model_fix(batch_linear_vel, batch_angular_vel)
    
    px_list_a = []
    pz_list_a = []
    for px_v in px_list:
        px_list_a.append(px_v.unsqueeze(1))
    for pz_v in pz_list:
        pz_list_a.append(pz_v.unsqueeze(1))        
    batch_px_list = torch.cat(px_list_a, axis=1)
    batch_pz_list = torch.cat(pz_list_a, axis=1)
    last_yaw = to_numpy(ry_list[-1])
        
    wandb_list = []

    for i in range(num_images_log):
        fig = plt.figure(figsize=(34, 16), dpi=80)
        gs = fig.add_gridspec(2,3)
        ax_graph = fig.add_subplot(gs[0:1, 0:1])
        ax_graph2 = fig.add_subplot(gs[1:2, 0:1])        
        ax_ob = fig.add_subplot(gs[0:1, 1:2])
        ax_goal = fig.add_subplot(gs[0:1, 2:3])
        ax_past = fig.add_subplot(gs[1:2, 2:3])        
        ax_depth1 = fig.add_subplot(gs[1:2, 1:2])
                            
        x_seq_old = to_numpy(batch_px_list[i, 0:6])
        z_seq_old = to_numpy(batch_pz_list[i, 0:6])
        x_seq = to_numpy(batch_px_list[i, 6:14])
        z_seq = to_numpy(batch_pz_list[i, 6:14])
                        
        xgt = to_numpy(obj_poses[i,0])
        ygt = to_numpy(obj_poses[i,1])

        xest = to_numpy(last_poses[i,0])
        yest = to_numpy(last_poses[i,1])
        
        if (local_yaw[i].item()) % (2.0*3.14) > 3.14:
            ang_yaw = (local_yaw[i].item()) % (2.0*3.14) - 2.0*3.14
        else:
            ang_yaw = (local_yaw[i].item()) % (2.0*3.14)
        label = ang_yaw * 180 / 3.14
        label_action = last_yaw[i] * 180 / 3.14
        
        hup = 48#150
        hdown = 64#190
        bias = 10        
        #bias_x = 10
        batch_3d_point_flatten = to_numpy(batch_3d_point[i,:,hup:hdown,bias:416-bias]).reshape(3,-1)        
        batch_3d_point = batch_3d_point.cpu()
        mask1 = (batch_3d_point[i,1:2,:,:].cpu() < 0.0*torch.ones((1, 128, 416)))
        mask1_x = (batch_3d_point[i,1:2,:,:].cpu() < 0.0*torch.ones((1, 128, 416)))        
        mask2 = (batch_3d_point[i,1:2,:,:].cpu() > -0.3*torch.ones((1, 128, 416)))    
        mask2_x = (batch_3d_point[i,1:2,:,:].cpu() > -0.3*torch.ones((1, 128, 416)))                    
        mask = torch.logical_and(mask1, mask2)[:,:,bias:416-bias]

        ped_x = ped_list[i,0:8].detach().numpy()                      
        ped_y = ped_list[i,8:16].detach().numpy()  
         
        robot_x = robot_list[i,0:8].detach().numpy()                      
        robot_y = robot_list[i,8:16].detach().numpy()          
                      
        pedest_x = est_ped_traj[i,0:8].detach().numpy()                      
        pedest_z = est_ped_traj[i,8:16].detach().numpy()  
        pedest_zero_x = est_ped_traj_zeros[i,0:8].detach().numpy()
        pedest_zero_z = est_ped_traj_zeros[i,8:16].detach().numpy() 
                                           
        xenv = to_numpy(batch_3d_point[i,0:1,:,bias:416-bias][mask]).reshape(-1)
        zenv = to_numpy(batch_3d_point[i,2:3,:,bias:416-bias][mask]).reshape(-1)  

        ax_graph.plot(x_seq_old, z_seq_old, marker = 's', color='cyan')                
        ax_graph.plot(x_seq, z_seq, marker = 'o', color='blue')
        ax_graph.plot(ped_x, ped_y, marker = 'o', color='magenta')
        ax_graph.plot(ped_x[0], ped_y[0], marker = 's', color='red')   
        ax_graph.plot(pedest_zero_x, pedest_zero_z, marker = 'o', color='cyan')  
        ax_graph.plot(pedest_x, pedest_z, marker = 'o', color='orange')              
          
        ax_graph.plot(robot_x, robot_y, marker = 'o', color='green')
        ax_graph.plot(robot_x[0], robot_y[0], marker = 's', color='green')                          
        ax_graph.plot(-ygt, xgt, marker = '*', color='red')
        ax_graph.plot(-yest, xest, marker = '+', color='green')                   
        ax_graph.scatter(xenv, zenv, marker = '.', color='black')     
        ax_graph.annotate(str(label)+' degrees', xy = (-ygt, xgt), xytext = (-20, 20),textcoords = 'offset points')

        ax_graph2.plot(x_seq_old, z_seq_old, marker = 's', color='cyan')            
        ax_graph2.plot(x_seq, z_seq, marker = 'o', color='blue')
        ax_graph2.plot(ped_x, ped_y, marker = 'o', color='magenta')
        ax_graph2.plot(ped_x[0], ped_y[0], marker = 's', color='red')           
        ax_graph2.plot(pedest_zero_x, pedest_zero_z, marker = 'o', color='cyan')  
        ax_graph2.plot(pedest_x, pedest_z, marker = 'o', color='orange')          
        ax_graph2.plot(robot_x, robot_y, marker = 'o', color='green')
        ax_graph2.plot(robot_x[0], robot_y[0], marker = 's', color='green')                                         
        ax_graph2.plot(-ygt, xgt, marker = '*', color='red')
        ax_graph2.plot(-yest, xest, marker = '+', color='green')          
        ax_graph2.scatter(xenv, zenv, marker = '.', color='black')
        ax_graph2.annotate(str(label)+' degrees (GT)', xy = (0.0, 0.0), xytext = (-20, 20),textcoords = 'offset points')        
        ax_graph2.annotate(str(label_action)+' degrees', xy = (-0.0, 0.0), xytext = (-20, 00),textcoords = 'offset points')
        ang_vel = (180.0*0.333*torch.sum(angular_vel[i,:])/3.1415).cpu().detach().numpy()
        ax_graph2.annotate(str(ang_vel)+' degrees (Vel)', xy = (-0.0, 0.0), xytext = (-20, -20),textcoords = 'offset points')
                                                        
        for j in range(8):
            circle = plt.Circle((x_seq[j], z_seq[j]), to_numpy(rsize)[i,0,0], color='black', fill=False)
            ax_graph2.add_patch(circle)          
        
        ax_past.plot(linear_vel[i,:].cpu().detach().numpy(), marker = 'o', color='red')
        ax_past.plot(angular_vel[i,:].cpu().detach().numpy(), marker = 'o', color='blue')
                                              
        obs_image = to_numpy(batch_viz_obs_images[i])
        obs_image = np.moveaxis(obs_image, 0, -1)        
        obs_image_past = to_numpy(batch_viz_obs_images_past[i])
        obs_image_past = np.moveaxis(obs_image_past, 0, -1)             
        goal_image = to_numpy(batch_viz_goal_images[i])
        ax_ob.imshow((255.0*obs_image).astype(np.uint8))              
        goal_image = np.moveaxis(goal_image, 0, -1)
        ax_goal.imshow((255.0*goal_image).astype(np.uint8))        
            
        ax_depth1.imshow(to_numpy(batch_3d_point[i,2,:,:]).astype(np.uint8), cmap='jet', interpolation='nearest')
                        
        # set title
        ax_graph.set_title(f"est. trajectory")
        ax_graph.set_xlim(-10, 10)
        ax_graph.set_ylim(-1, 10)       
        ax_graph2.set_title(f"est. trajectory")
        ax_graph2.set_xlim(-5.0, 5.0)
        ax_graph2.set_ylim(-5.0, 5.0)            
        ax_ob.set_title(f"observation")
        ax_goal.set_title(f"goal image")
        ax_past.set_title(f"velocity command")
                        
        # make the plot large
        fig.set_size_inches(18.5, 10.5)
        
        save_path = os.path.join(visualize_path, f"sample_ped_{i}.png")
        plt.savefig(save_path)
        wandb_list.append(wandb.Image(save_path))            
        save_path = os.path.join(visualize_path, f"sample_{i}.png")
        plt.savefig(save_path)            
            
        plt.close(fig)

def visualize_il_estimation(
    batch_viz_obs_images: torch.Tensor,
    batch_viz_obs_images_past: torch.Tensor,    
    batch_viz_goal_images: torch.Tensor,
    obj_poses: torch.Tensor,
    local_yaw: torch.Tensor,
    action_pred: torch.Tensor,
    eval_type: str,    
    project_folder: str,
    epoch: int,
    num_images_log: int,
    num_samples: int = 30,    
    use_wandb: bool = True,
):
    """Plot samples from the exploration model."""

    #print(project_folder, eval_type)
    visualize_path = os.path.join(
        project_folder,
        "visualize",
        eval_type,
        f"epoch{epoch}",
        "action_sampling_prediction",
    )        
    
    if not os.path.isdir(visualize_path):
        os.makedirs(visualize_path)

    num_images_log = min(num_images_log, batch_viz_obs_images.shape[0], batch_viz_goal_images.shape[0], obj_poses.shape[0])    
    metric_waypoint_spacing = 0.3 #normalization   
        
    wandb_list = []

    for i in range(num_images_log):
        fig = plt.figure(figsize=(34, 16), dpi=80)
        gs = fig.add_gridspec(2,3)
        ax_graph = fig.add_subplot(gs[0:2, 0:1])      
        ax_ob = fig.add_subplot(gs[0:1, 1:2])
        ax_goal = fig.add_subplot(gs[0:1, 2:3])
        ax_past = fig.add_subplot(gs[1:2, 2:3])        
        ax_depth1 = fig.add_subplot(gs[1:2, 1:2])
                            
        xgt = to_numpy(obj_poses[i,0])
        ygt = to_numpy(obj_poses[i,1])
        
        if (local_yaw[i].item()) % (2.0*3.14) > 3.14:
            ang_yaw = (local_yaw[i].item()) % (2.0*3.14) - 2.0*3.14
        else:
            ang_yaw = (local_yaw[i].item()) % (2.0*3.14)
        label = ang_yaw * 180 / 3.14        
        label_action = torch.atan(action_pred[i, -1, 3]/action_pred[i, -1, 2]) * 180 / 3.14
        
        x_seq = action_pred[i, :, 0].detach().cpu().numpy()*metric_waypoint_spacing
        y_seq = action_pred[i, :, 1].detach().cpu().numpy()*metric_waypoint_spacing
        
                
        ax_graph.plot(-y_seq, x_seq, marker = 'o', color='blue')                       
        ax_graph.plot(-ygt, xgt, marker = '*', color='red')                
        ax_graph.annotate(str(label)+' degrees (GT)', xy = (0.0, 0.0), xytext = (-20, 20),textcoords = 'offset points')        
        ax_graph.annotate(str(label_action)+' degrees', xy = (-0.0, 0.0), xytext = (-20, 00),textcoords = 'offset points')   
        
        ax_past.plot(x_seq, marker = 'o', color='red')
        ax_past.plot(y_seq, marker = 'o', color='blue')
                                              
        obs_image = to_numpy(batch_viz_obs_images[i])
        obs_image = np.moveaxis(obs_image, 0, -1)                    
        goal_image = to_numpy(batch_viz_goal_images[i])
        goal_image = np.moveaxis(goal_image, 0, -1)        
        ax_ob.imshow((255.0*obs_image).astype(np.uint8))               
        ax_goal.imshow((255.0*goal_image).astype(np.uint8))        
                        
        # set title
        ax_graph.set_title(f"est. trajectory")               
        ax_ob.set_title(f"observation")
        ax_goal.set_title(f"goal image")
        ax_past.set_title(f"velocity command")
                        
        # make the plot large
        fig.set_size_inches(18.5, 10.5)       
        save_path = os.path.join(visualize_path, f"sample_ped_{i}.png")
        plt.savefig(save_path)
        wandb_list.append(wandb.Image(save_path))        
        plt.close(fig)

def visualize_il_estimation_gps(
    batch_viz_obs_images: torch.Tensor,
    batch_viz_obs_images_past: torch.Tensor,    
    batch_viz_goal_images: torch.Tensor,
    obj_poses: torch.Tensor,
    local_yaw: torch.Tensor,
    action_pred: torch.Tensor,
    action: torch.Tensor,
    eval_type: str,    
    project_folder: str,
    epoch: int,
    num_images_log: int,
    num_samples: int = 30,    
    use_wandb: bool = True,
):
    """Plot samples from the exploration model."""

    visualize_path = os.path.join(
        project_folder,
        "visualize",
        eval_type,
        f"epoch{epoch}",
        "action_sampling_prediction",
    )        
    
    if not os.path.isdir(visualize_path):
        os.makedirs(visualize_path)

    num_images_log = min(num_images_log, batch_viz_obs_images.shape[0], batch_viz_goal_images.shape[0], obj_poses.shape[0])    
    metric_waypoint_spacing = 1.0 #normalization   
        
    wandb_list = []

    for i in range(num_images_log):
        fig = plt.figure(figsize=(34, 16), dpi=80)
        gs = fig.add_gridspec(2,3)
        ax_graph = fig.add_subplot(gs[0:2, 0:1])      
        ax_ob = fig.add_subplot(gs[0:1, 1:2])
        ax_goal = fig.add_subplot(gs[0:1, 2:3])
        ax_past = fig.add_subplot(gs[1:2, 2:3])        
        ax_yaw = fig.add_subplot(gs[1:2, 1:2])
                            
        xgt = to_numpy(obj_poses[i,0])*metric_waypoint_spacing
        ygt = to_numpy(obj_poses[i,1])*metric_waypoint_spacing
        
        if (local_yaw[i].item()) % (2.0*3.14) > 3.14:
            ang_yaw = (local_yaw[i].item()) % (2.0*3.14) - 2.0*3.14
        else:
            ang_yaw = (local_yaw[i].item()) % (2.0*3.14)
        label = ang_yaw * 180 / 3.14        
        label_action = torch.atan(action_pred[i, -1, 3]/action_pred[i, -1, 2]) * 180 / 3.14
        
        yaw_seq = torch.atan2(action_pred[i, :, 3], action_pred[i, :, 2]).detach().cpu().numpy() * 180 / 3.14
        yaw_label = torch.atan2(action[i, :, 3], action[i, :, 2]).detach().cpu().numpy() * 180 / 3.14
        
        x_seq = action_pred[i, :, 0].detach().cpu().numpy()*metric_waypoint_spacing
        y_seq = action_pred[i, :, 1].detach().cpu().numpy()*metric_waypoint_spacing
        x_label = action[i, :, 0].detach().cpu().numpy()*metric_waypoint_spacing
        y_label = action[i, :, 1].detach().cpu().numpy()*metric_waypoint_spacing        
                
        ax_graph.plot(-y_seq, x_seq, marker = 'o', color='blue')        
        ax_graph.plot(-y_label, x_label, marker = 'o', color='red')                            
        ax_graph.plot(-ygt, xgt, marker = '*', markersize=15, color='red')                
        ax_graph.annotate(str(label)+' degrees (GT)', xy = (0.0, 0.0), xytext = (-20, 20),textcoords = 'offset points')        
        ax_graph.annotate(str(label_action)+' degrees', xy = (-0.0, 0.0), xytext = (-20, 00),textcoords = 'offset points')   
        
        ax_past.plot(x_seq, marker = 'o', color='red')
        ax_past.plot(y_seq, marker = 'o', color='blue')
        ax_past.plot(x_label, marker = 's', color='red')
        ax_past.plot(y_label, marker = 's', color='blue')
                        
        ax_yaw.plot(yaw_seq, marker = 'o', color='blue')
        ax_yaw.plot(yaw_label, marker = 's', color='blue')
                                              
        obs_image = to_numpy(batch_viz_obs_images[i])
        obs_image = np.moveaxis(obs_image, 0, -1)                    
        goal_image = to_numpy(batch_viz_goal_images[i])
        goal_image = np.moveaxis(goal_image, 0, -1)        
        ax_ob.imshow((255.0*obs_image).astype(np.uint8))               
        ax_goal.imshow((255.0*goal_image).astype(np.uint8))        
                        
        # set title
        ax_graph.set_title(f"est. trajectory")                
        ax_ob.set_title(f"observation")
        ax_goal.set_title(f"goal image")
        ax_past.set_title(f"velocity command")
                        
        # make the plot large
        fig.set_size_inches(18.5, 10.5)
        
        save_path = os.path.join(visualize_path, f"sample_ped_{i}.png")
        plt.savefig(save_path)
        wandb_list.append(wandb.Image(save_path))        
        plt.close(fig)

def visualize_il2_estimation_map(
    batch_viz_obs_images: torch.Tensor,
    batch_viz_obs_images_past: torch.Tensor,    
    batch_viz_goal_images: torch.Tensor,
    batch_viz_cur_map: torch.Tensor,
    batch_viz_goal_map: torch.Tensor,    
    obj_poses: torch.Tensor,
    local_yaw: torch.Tensor,
    action_pred: torch.Tensor,
    action_label: torch.Tensor,    
    action_origin: torch.Tensor,  
    mask_number: torch.Tensor,
    eval_type: str,    
    project_folder: str,
    epoch: int,
    num_images_log: int,
    num_samples: int = 30,    
    use_wandb: bool = True,
):
    """Plot samples from the exploration model."""

    visualize_path = os.path.join(
        project_folder,
        "visualize",
        eval_type,
        f"epoch{epoch}",
        "action_sampling_prediction",
    )        
    
    if not os.path.isdir(visualize_path):
        os.makedirs(visualize_path)

    num_images_log = min(num_images_log, batch_viz_obs_images.shape[0], batch_viz_goal_images.shape[0], obj_poses.shape[0])    
    metric_waypoint_spacing = 0.25 #normalization   
        
    wandb_list = []

    for i in range(num_images_log):
        fig = plt.figure(figsize=(34, 16), dpi=80)
        gs = fig.add_gridspec(2,3)
        ax_graph = fig.add_subplot(gs[0:1, 0:1])      
        ax_ob = fig.add_subplot(gs[0:1, 1:2])
        ax_goal = fig.add_subplot(gs[0:1, 2:3])
        ax_past = fig.add_subplot(gs[1:2, 0:1])        
        ax_curmap = fig.add_subplot(gs[1:2, 1:2])
        ax_goalmap = fig.add_subplot(gs[1:2, 2:3])
                            
        xgt = to_numpy(obj_poses[i,0])
        ygt = to_numpy(obj_poses[i,1])
        
        if (local_yaw[i].item()) % (2.0*3.14) > 3.14:
            ang_yaw = (local_yaw[i].item()) % (2.0*3.14) - 2.0*3.14
        else:
            ang_yaw = (local_yaw[i].item()) % (2.0*3.14)
        label = ang_yaw * 180 / 3.14        
        label_action = torch.atan(action_pred[i, -1, 3]/action_pred[i, -1, 2]) * 180 / 3.14
        
        mask_type = mask_number[i]        
        x_seq = action_pred[i, :, 0].detach().cpu().numpy()*metric_waypoint_spacing
        y_seq = action_pred[i, :, 1].detach().cpu().numpy()*metric_waypoint_spacing
        
        x_seq_l = action_label[i, :, 0].detach().cpu().numpy()*metric_waypoint_spacing*0.5
        y_seq_l = action_label[i, :, 1].detach().cpu().numpy()*metric_waypoint_spacing*0.5
        x_seq_o = action_origin[i, :, 0].detach().cpu().numpy()*metric_waypoint_spacing
        y_seq_o = action_origin[i, :, 1].detach().cpu().numpy()*metric_waypoint_spacing
                                
        ax_graph.plot(-y_seq, x_seq, marker = 'o', color='blue', label="est")        
        ax_graph.plot(-y_seq_l, x_seq_l, marker = 'o', color='red', label="label")               
        ax_graph.plot(-y_seq_o, x_seq_o, marker = 'o', color='magenta', label="original label")               
        ax_graph.plot(-ygt, xgt, marker = '*', color='red')                
        ax_graph.annotate(str(label)+' degrees (GT)', xy = (0.0, 0.0), xytext = (-20, 20),textcoords = 'offset points')  
        ax_graph.annotate("X:" + str(xgt) + "Y:" + str(ygt), xy = (-0.0, 0.0), xytext = (-20, 00),textcoords = 'offset points')         
        if mask_type == 0:
            ax_graph.annotate("pose only", xy = (-0.0, 0.0), xytext = (-20, -20),textcoords = 'offset points')   
        elif mask_type == 1:
            ax_graph.annotate("satellite only", xy = (-0.0, 0.0), xytext = (-20, -20),textcoords = 'offset points')           
        else:
            ax_graph.annotate("pose and satellite", xy = (-0.0, 0.0), xytext = (-20, -20),textcoords = 'offset points')    
                    
        ax_past.plot(x_seq, marker = 'o', color='red')
        ax_past.plot(y_seq, marker = 'o', color='blue')
                                              
        obs_image = to_numpy(batch_viz_obs_images[i])
        obs_image = np.moveaxis(obs_image, 0, -1)                    
        goal_image = to_numpy(batch_viz_goal_images[i])
        goal_image = np.moveaxis(goal_image, 0, -1)        
        ax_ob.imshow((255.0*obs_image).astype(np.uint8))               
        ax_goal.imshow((255.0*goal_image).astype(np.uint8))        

        map_image_cur = to_numpy(batch_viz_cur_map[i])
        map_image_cur = np.moveaxis(map_image_cur, 0, -1)                    
        map_image_goal = to_numpy(batch_viz_goal_map[i])
        map_image_goal = np.moveaxis(map_image_goal, 0, -1)        
        ax_curmap.imshow((255.0*map_image_cur).astype(np.uint8))               
        ax_goalmap.imshow((255.0*map_image_goal).astype(np.uint8))     
                        
        # set title
        ax_graph.set_title(f"est. trajectory")
        ax_graph.set_xlim(-2.0, 2.0)
        ax_graph.set_ylim(-1.0, 3.0)
        ax_graph.legend(loc='best')                  
        ax_ob.set_title(f"observation")
        ax_goal.set_title(f"goal image")
        ax_past.set_title(f"velocity command")
                        
        # make the plot large
        fig.set_size_inches(18.5, 10.5)
        
        save_path = os.path.join(visualize_path, f"sample_ped_{i}.png")
        plt.savefig(save_path)
        wandb_list.append(wandb.Image(save_path))        
        plt.close(fig)



def visualize_il2_estimation(
    batch_viz_obs_images: torch.Tensor,
    batch_viz_obs_images_past: torch.Tensor,    
    batch_viz_goal_images: torch.Tensor,
    obj_poses: torch.Tensor,
    local_yaw: torch.Tensor,
    action_pred: torch.Tensor,
    action_label: torch.Tensor,    
    action_origin: torch.Tensor,  
    eval_type: str,    
    project_folder: str,
    epoch: int,
    num_images_log: int,
    num_samples: int = 30,    
    use_wandb: bool = True,
):
    """Plot samples from the exploration model."""

    visualize_path = os.path.join(
        project_folder,
        "visualize",
        eval_type,
        f"epoch{epoch}",
        "action_sampling_prediction",
    )        
    
    if not os.path.isdir(visualize_path):
        os.makedirs(visualize_path)

    num_images_log = min(num_images_log, batch_viz_obs_images.shape[0], batch_viz_goal_images.shape[0], obj_poses.shape[0])    
    metric_waypoint_spacing = 0.25 #normalization   
        
    wandb_list = []

    for i in range(num_images_log):
        fig = plt.figure(figsize=(34, 16), dpi=80)
        gs = fig.add_gridspec(2,3)
        ax_graph = fig.add_subplot(gs[0:2, 0:1])      
        ax_ob = fig.add_subplot(gs[0:1, 1:2])
        ax_goal = fig.add_subplot(gs[0:1, 2:3])
        ax_past = fig.add_subplot(gs[1:2, 2:3])        
        ax_depth1 = fig.add_subplot(gs[1:2, 1:2])
                            
        xgt = to_numpy(obj_poses[i,0])
        ygt = to_numpy(obj_poses[i,1])
        
        if (local_yaw[i].item()) % (2.0*3.14) > 3.14:
            ang_yaw = (local_yaw[i].item()) % (2.0*3.14) - 2.0*3.14
        else:
            ang_yaw = (local_yaw[i].item()) % (2.0*3.14)
        label = ang_yaw * 180 / 3.14        
        label_action = torch.atan(action_pred[i, -1, 3]/action_pred[i, -1, 2]) * 180 / 3.14
        
        x_seq = action_pred[i, :, 0].detach().cpu().numpy()*metric_waypoint_spacing
        y_seq = action_pred[i, :, 1].detach().cpu().numpy()*metric_waypoint_spacing
        
        x_seq_l = action_label[i, :, 0].detach().cpu().numpy()*metric_waypoint_spacing*0.5
        y_seq_l = action_label[i, :, 1].detach().cpu().numpy()*metric_waypoint_spacing*0.5
        x_seq_o = action_origin[i, :, 0].detach().cpu().numpy()*metric_waypoint_spacing
        y_seq_o = action_origin[i, :, 1].detach().cpu().numpy()*metric_waypoint_spacing
                                
        ax_graph.plot(-y_seq, x_seq, marker = 'o', color='blue', label="est")        
        ax_graph.plot(-y_seq_l, x_seq_l, marker = 'o', color='red', label="label")               
        ax_graph.plot(-y_seq_o, x_seq_o, marker = 'o', color='magenta', label="original label")               
        ax_graph.plot(-ygt, xgt, marker = '*', color='red')                
        ax_graph.annotate(str(label)+' degrees (GT)', xy = (0.0, 0.0), xytext = (-20, 20),textcoords = 'offset points')        
        ax_graph.annotate(str(label_action)+' degrees', xy = (-0.0, 0.0), xytext = (-20, 00),textcoords = 'offset points')   
        
        ax_past.plot(x_seq, marker = 'o', color='red')
        ax_past.plot(y_seq, marker = 'o', color='blue')
                                              
        obs_image = to_numpy(batch_viz_obs_images[i])
        obs_image = np.moveaxis(obs_image, 0, -1)                    
        goal_image = to_numpy(batch_viz_goal_images[i])
        goal_image = np.moveaxis(goal_image, 0, -1)        
        ax_ob.imshow((255.0*obs_image).astype(np.uint8))               
        ax_goal.imshow((255.0*goal_image).astype(np.uint8))        
                        
        # set title
        ax_graph.set_title(f"est. trajectory")
        ax_graph.set_xlim(-2.0, 2.0)
        ax_graph.set_ylim(-1.0, 3.0)
        ax_graph.legend(loc='best')                  
        ax_ob.set_title(f"observation")
        ax_goal.set_title(f"goal image")
        ax_past.set_title(f"velocity command")
                        
        # make the plot large
        fig.set_size_inches(18.5, 10.5)
        
        save_path = os.path.join(visualize_path, f"sample_ped_{i}.png")
        plt.savefig(save_path)
        wandb_list.append(wandb.Image(save_path))        
        plt.close(fig)


def visualize_lelan_estimation(
    batch_viz_obs_images: torch.Tensor,
    batch_viz_goal_images: torch.Tensor,
    obj_poses: torch.Tensor,
    obj_inst: torch.Tensor,
    linear_vel: torch.Tensor,
    angular_vel: torch.Tensor,
    last_poses: torch.Tensor,
    eval_type: str,    
    project_folder: str,
    epoch: int,
    num_images_log: int,
    num_samples: int = 30,    
    use_wandb: bool = True,
):
    """Plot samples from the exploration model."""

    visualize_path = os.path.join(
        project_folder,
        "visualize",
        eval_type,
        f"epoch{epoch}",
        "action_sampling_prediction",
    )
    if not os.path.isdir(visualize_path):
        os.makedirs(visualize_path)

    num_images_log = min(num_images_log, batch_viz_obs_images.shape[0], batch_viz_goal_images.shape[0], obj_poses.shape[0], last_poses.shape[0])    
    batch_linear_vel = linear_vel[:num_images_log]
    batch_angular_vel = angular_vel[:num_images_log]
    
    px_list, pz_list, ry_list = robot_pos_model_fix(batch_linear_vel, batch_angular_vel)
    
    px_list_a = []
    pz_list_a = []
    for px_v in px_list:
        px_list_a.append(px_v.unsqueeze(1))
    for pz_v in pz_list:
        pz_list_a.append(pz_v.unsqueeze(1))        
    batch_px_list = torch.cat(px_list_a, axis=1)
    batch_pz_list = torch.cat(pz_list_a, axis=1)
    
    wandb_list = []

    for i in range(num_images_log):
        fig = plt.figure(figsize=(34, 16), dpi=80)
        gs = fig.add_gridspec(2,3)
        ax_graph = fig.add_subplot(gs[0:2, 0:1])
        ax_ob = fig.add_subplot(gs[0:1, 1:2])
        ax_goal = fig.add_subplot(gs[0:1, 2:3])
        ax_inst = fig.add_subplot(gs[1:2, 1:3])
                    
        x_seq = to_numpy(batch_px_list[i])
        z_seq = to_numpy(batch_pz_list[i])
                
        xgt = to_numpy(obj_poses[i,0])
        ygt = to_numpy(obj_poses[i,1])

        xest = to_numpy(last_poses[i,0])
        yest = to_numpy(last_poses[i,1])
        
        ax_graph.plot(x_seq, z_seq, marker = 'o', color='blue')
        ax_graph.plot(xgt, ygt, marker = '*', color='red')
        ax_graph.plot(xest, yest, marker = '+', color='green')
                
        obs_image = to_numpy(batch_viz_obs_images[i])
        prompt = obj_inst[i]
        goal_image = to_numpy(batch_viz_goal_images[i])
        # move channel to last dimension
        obs_image = np.moveaxis(obs_image, 0, -1)
        goal_image = np.moveaxis(goal_image, 0, -1)
        ax_ob.imshow(obs_image)
        ax_goal.imshow(goal_image)
        ax_inst.text(0, 0, prompt, fontsize = 12, color = 'black')
        ax_inst.axis('off')
                        
        # set title
        ax_graph.set_title(f"est. trajectory")
        ax_ob.set_title(f"observation")
        ax_goal.set_title(f"cropped goal image")
        
        # make the plot large
        fig.set_size_inches(18.5, 10.5)
        
        save_path = os.path.join(visualize_path, f"sample_{i}.png")
        plt.savefig(save_path)
        wandb_list.append(wandb.Image(save_path))
        plt.close(fig)
            
    if len(wandb_list) > 0 and use_wandb:
        wandb.log({f"{eval_type}_action_samples": wandb_list}, commit=False)       
        
def visualize_lelan_col_estimation(
    batch_viz_obs_images: torch.Tensor,
    batch_viz_goal_images: torch.Tensor,
    obj_poses: torch.Tensor,
    obj_inst: torch.Tensor,
    linear_vel: torch.Tensor,
    angular_vel: torch.Tensor,
    last_poses: torch.Tensor,
    ref_actions: torch.Tensor,
    eval_type: str,    
    project_folder: str,
    epoch: int,
    num_images_log: int,
    num_samples: int = 30,    
    use_wandb: bool = True,
):
    """Plot samples from the exploration model."""

    visualize_path = os.path.join(
        project_folder,
        "visualize",
        eval_type,
        f"epoch{epoch}",
        "action_sampling_prediction",
    )
    if not os.path.isdir(visualize_path):
        os.makedirs(visualize_path)

    num_images_log = min(num_images_log, batch_viz_obs_images.shape[0], batch_viz_goal_images.shape[0], obj_poses.shape[0], last_poses.shape[0])    
    batch_linear_vel = linear_vel[:num_images_log]
    batch_angular_vel = angular_vel[:num_images_log]
    
    px_list, pz_list, ry_list = robot_pos_model_fix(batch_linear_vel, batch_angular_vel)
    
    px_list_a = []
    pz_list_a = []
    for px_v in px_list:
        px_list_a.append(px_v.unsqueeze(1))
    for pz_v in pz_list:
        pz_list_a.append(pz_v.unsqueeze(1))        
    batch_px_list = torch.cat(px_list_a, axis=1)
    batch_pz_list = torch.cat(pz_list_a, axis=1)

    wandb_list = []
        
    for i in range(num_images_log):
        fig = plt.figure(figsize=(34, 16), dpi=80)
        gs = fig.add_gridspec(2,3)
        ax_graph = fig.add_subplot(gs[0:2, 0:1])
        ax_ob = fig.add_subplot(gs[0:1, 1:2])
        ax_goal = fig.add_subplot(gs[0:1, 2:3])
        ax_inst = fig.add_subplot(gs[1:2, 1:3])
                    
        x_seq = to_numpy(batch_px_list[i])
        z_seq = to_numpy(batch_pz_list[i])
                
        xgt = to_numpy(obj_poses[i,0])
        ygt = to_numpy(obj_poses[i,1])

        xest = to_numpy(last_poses[i,0])
        yest = to_numpy(last_poses[i,1])
        
        x_nomad = to_numpy(ref_actions[i,:,0])
        y_nomad = to_numpy(ref_actions[i,:,1])
        
        ax_graph.plot(x_seq, z_seq, marker = 'o', color='blue')
        ax_graph.plot(-y_nomad, x_nomad, marker = 'o', color='magenta')
        ax_graph.plot(xgt, ygt, marker = '*', color='red')
        ax_graph.plot(xest, yest, marker = '+', color='green')
                
        obs_image = to_numpy(batch_viz_obs_images[i])
        prompt = obj_inst[i]
        goal_image = to_numpy(batch_viz_goal_images[i])
        # move channel to last dimension
        obs_image = np.moveaxis(obs_image, 0, -1)
        goal_image = np.moveaxis(goal_image, 0, -1)
        ax_ob.imshow(obs_image)
        ax_goal.imshow(goal_image)
        ax_inst.text(0, 0, prompt, fontsize = 12, color = 'black')
        ax_inst.axis('off')
                        
        # set title
        ax_graph.set_title(f"est. trajectory")
        ax_ob.set_title(f"observation")
        ax_goal.set_title(f"cropped goal image")
        
        # make the plot large
        fig.set_size_inches(18.5, 10.5)
        
        save_path = os.path.join(visualize_path, f"sample_{i}.png")
        plt.savefig(save_path)
        wandb_list.append(wandb.Image(save_path))
        plt.close(fig)
            
    if len(wandb_list) > 0 and use_wandb:
        wandb.log({f"{eval_type}_action_samples": wandb_list}, commit=False)           
        
def calculate_fov(K, image_size):
    # Extract focal lengths from the intrinsic matrix
    f_x = K[0, 0]
    f_y = K[1, 1]
    
    # Extract image dimensions
    W, H = image_size
    
    # Calculate horizontal and vertical field of view
    hfov = 2 * np.arctan(W / (2 * f_x)) * (180 / np.pi)  # Convert to degrees
    vfov = 2 * np.arctan(H / (2 * f_y)) * (180 / np.pi)  # Convert to degrees
    
    return hfov, vfov
