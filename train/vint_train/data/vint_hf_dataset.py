import sys

from enum import Enum
import numpy as np
from typing import Tuple, Callable
from pathlib import Path
import einops
import zarr

import torch
import torch.utils.data
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import (
    load_previous_and_future_frames,
)
from lerobot.common.datasets.video_utils import load_from_videos

from typing import Iterator
import random
from tqdm import tqdm
import pickle

#from ViNT
def yaw_rotmat(yaw: float | np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    if isinstance(yaw, torch.Tensor):
        return torch.tensor(
            [
                [torch.cos(yaw), -torch.sin(yaw), torch.zeros_like(yaw)],
                [torch.sin(yaw), torch.cos(yaw), torch.zeros_like(yaw)],
                [torch.zeros_like(yaw), torch.zeros_like(yaw), torch.ones_like(yaw)],
            ],
        )
    else:
        return np.array(
            [
                [np.cos(yaw), -np.sin(yaw), 0.0],
                [np.sin(yaw), np.cos(yaw), 0.0],
                [0.0, 0.0, 1.0],
            ],
        )
        
def trans_mat(pos: float | np.ndarray | torch.Tensor, yaw: float | np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    if isinstance(yaw, torch.Tensor):
        return torch.tensor(
            [
                [torch.cos(yaw), -torch.sin(yaw), pos[0]],
                [torch.sin(yaw), torch.cos(yaw), pos[1]],
                [torch.zeros_like(yaw), torch.zeros_like(yaw), torch.ones_like(yaw)],
            ],
        )
    else:
        return np.array(
            [
                [np.cos(yaw), -np.sin(yaw), pos[0]],
                [np.sin(yaw), np.cos(yaw), pos[1]],
                [0.0, 0.0, 1.0],
            ],
        )

def to_local_coords(
    positions: np.ndarray | torch.Tensor, curr_pos: np.ndarray | torch.Tensor, curr_yaw: float | np.ndarray | torch.Tensor
) -> np.ndarray | torch.Tensor:
    """
    Convert positions to local coordinates

    Args:
        positions (np.ndarray): positions to convert
        curr_pos (np.ndarray): current position
        curr_yaw (float): current yaw
    Returns:
        np.ndarray: positions in local coordinates
    """
    rotmat = yaw_rotmat(curr_yaw)
    if positions.shape[-1] == 2:
        rotmat = rotmat[:2, :2]
    elif positions.shape[-1] == 3:
        pass
    else:
        raise ValueError

    return (positions - curr_pos) @ rotmat
    
def to_local_coords_yaw(
    positions: np.ndarray | torch.Tensor, curr_pos: np.ndarray | torch.Tensor, curr_yaw: float | np.ndarray | torch.Tensor,  goal_yaw: float | np.ndarray | torch.Tensor
) -> np.ndarray | torch.Tensor:
    """
    Convert positions to local coordinates

    Args:
        positions (np.ndarray): positions to convert
        curr_pos (np.ndarray): current position
        curr_yaw (float): current yaw
    Returns:
        np.ndarray: positions in local coordinates
    """
    cur_mat = trans_mat(curr_pos, curr_yaw)
    goal_mat = trans_mat(positions[0], goal_yaw)    
    cur_mat_inv = torch.linalg.inv(cur_mat)
    relative_mat = torch.matmul(cur_mat_inv, goal_mat)

    return relative_mat
    
#from ViNT end    

class ActionFormat(Enum):
    WAYPOINT = 1
    WAYPOINT_ANGLE = 2
    LINEAR_ANGULAR = 3

    def __str__(self):
        return self.name.lower()

    @staticmethod
    def from_str(s: str) -> "ActionFormat":
        return ActionFormat[s.upper()]

def load_pickle(
    dataset: zarr.Array,
    index: int,
    episode_data_index: dict[str, torch.Tensor],  
    delta_timestamps: dict[str, list[float]],      
    ) -> dict[torch.Tensor]:
    #
    ep_id = dataset["episode_index"][index].item()
    ep_data_id_from = episode_data_index["from"][ep_id].item()
    ep_data_id_to = episode_data_index["to"][ep_id].item()
    ep_data_ids = torch.arange(ep_data_id_from, ep_data_id_to, 1)    
    
    for key, delta_ts in delta_timestamps.items():
        current_ts = dataset["timestamp"][index]
        query_ts = current_ts + torch.tensor(delta_ts)
        ep_timestamps = torch.from_numpy(dataset["timestamp"][ep_data_id_from:ep_data_id_to]).float()
        dist = torch.cdist(query_ts[:, None], ep_timestamps[:, None], p=1)
        min_, argmin_ = dist.min(1)
        data_ids = ep_data_ids[argmin_].numpy() 
    return data_ids
                
def load_frames_zarr(
    dataset: zarr.Array,
    index: int,
    episode_data_index: dict[str, torch.Tensor],
    delta_timestamps: dict[str, list[float]],
    tolerance_s: float,
) -> dict[torch.Tensor]:
    # get indices of the frames associated to the episode, and their timestamps
    ep_id = dataset["episode_index"][index].item()
    ep_data_id_from = episode_data_index["from"][ep_id].item()
    ep_data_id_to = episode_data_index["to"][ep_id].item()
    ep_data_ids = torch.arange(ep_data_id_from, ep_data_id_to, 1)

    # load timestamps
    ep_timestamps = torch.from_numpy(dataset["timestamp"][ep_data_id_from:ep_data_id_to]).float()

    # we make the assumption that the timestamps are sorted
    ep_first_ts = ep_timestamps[0]
    ep_last_ts = ep_timestamps[-1]
    current_ts = dataset["timestamp"][index]

    item = {}

    for key, delta_ts in delta_timestamps.items():
        # if it is a video frame
        timestamp_key = f"{key}.timestamp"
        path_key = f"{key}.path"
        is_video = timestamp_key in dataset.keys() and path_key in dataset.keys()

        # get timestamps used as query to retrieve data of previous/future frames
        if delta_ts is None:
            if key in dataset.keys():
                item[key] = torch.from_numpy(np.asarray(dataset[key][index]))
            elif is_video:
                item[key] = [
                    {"path": dataset[path_key][i.item()], "timestamp": dataset[timestamp_key][i.item()]}
                    for i in ep_data_ids
                ]
            else:
                raise ValueError(f"Timestamp key {timestamp_key} not found in dataset")
        else:
            query_ts = current_ts + torch.tensor(delta_ts)

            # compute distances between each query timestamp and all timestamps of all the frames belonging to the episode                               
            dist = torch.cdist(query_ts[:, None], ep_timestamps[:, None], p=1)
            min_, argmin_ = dist.min(1)

            # TODO(rcadene): synchronize timestamps + interpolation if needed

            is_pad = min_ > tolerance_s
            assert ((query_ts[is_pad] < ep_first_ts) | (ep_last_ts < query_ts[is_pad])).all(), (
                f"One or several timestamps unexpectedly violate the tolerance ({min_} > {tolerance_s=}) inside episode range."
                "This might be due to synchronization issues with timestamps during data collection."
            )

            # get dataset indices corresponding to frames to be loaded
            data_ids = ep_data_ids[argmin_].numpy()

            if is_video:
                # video mode where frame are expressed as dict of path and timestamp
                item[key] = [
                    {"path": dataset[path_key][i], "timestamp": float(dataset[timestamp_key][i])}
                    for i in data_ids
                ]
            else:
                item[key] = torch.from_numpy(dataset[key][data_ids])

            item[f"{key}_is_pad"] = is_pad

    return item


class ViNTLeRobotDataset(LeRobotDataset):
    def __init__(
        self,
        repo_id: str,
        video: str,
        root: Path | None,
        split: str = "train",
        action_format: ActionFormat | str = ActionFormat.WAYPOINT,
        action_horizon: int = 8,
        action_spacing: int = 1,
        goal_horizon: int = 20,
        context_size: int = 5,
        context_spacing: int = 1,
        dataset_framerate: int = 10,
        image_size: Tuple[int, int] = (120, 160),
        image_transforms: Callable | None = None,
        sacson: bool = False
    ):
        """
        Main ViNT dataset class
        """
        if isinstance(action_format, str):
            action_format = ActionFormat.from_str(action_format)
        self.action_format = action_format

        if action_format == ActionFormat.WAYPOINT:
            self.num_action_params = 2
        elif action_format == ActionFormat.WAYPOINT_ANGLE:
            self.num_action_params = 3
        elif action_format == ActionFormat.LINEAR_ANGULAR:
            self.num_action_params = 2

        self.dt = 1 / dataset_framerate
        self.action_spacing = action_spacing
        self.action_horizon = action_horizon
        self.goal_horizon = goal_horizon
        self.context_size = context_size
        self.context_spacing = context_spacing
        self.image_size = image_size
        self.sacson = sacson

        print("root_init", root)
        super().__init__(
            repo_id=repo_id,
            video=video,
            root=root,
            split=split,
            image_transforms=image_transforms,
            delta_timestamps={
                "observation.filtered_position": [0.0],
                "observation.relative_position": [0.0],
                "observation.filtered_heading": [0.0],
                "observation.images.front": [i * context_spacing * self.dt for i in range(-context_size, 1)],
                "action": [i * action_spacing * self.dt for i in range(action_horizon)],                
            },
        )

        # Build a cache of episode data indices
        self.dataset_cache = zarr.load(Path(root) / "frodobots_dataset" / "dataset_cache.zarr")
        self.dataset_cache = {
            k: np.asarray(v) for k, v in self.dataset_cache.items()
        }

    def _image_transforms(self, img: torch.Tensor, flip) -> torch.Tensor:
        """
        Args:
            img (torch.Tensor): image tensor
        Returns:
            torch.Tensor: transformed image
        """
        if self.image_transforms is not None:
            img = self.image_transforms(img)

        original_height, original_width = img.shape[-2:]
        target_aspect = 4 / 3
        img = TF.resize(img, self.image_size)
        if flip:
            img = torch.flip(img, dims=(-1,))
        return img

    def _image_transforms_depth(self, img: torch.Tensor, flip) -> torch.Tensor:
        """
        Args:
            img (torch.Tensor): image tensor
        Returns:
            torch.Tensor: transformed image
        """

        img_rsize = TF.resize(img, (128, 416)) #
        if flip:
            img_rsize = torch.flip(img_rsize, dims=(-1,))
            img = torch.flip(img, dims=(-1,))                      
        return img_rsize, img

    def viz_rollout(self, actions: torch.Tensor) -> torch.Tensor:
        if self.action_format == ActionFormat.WAYPOINT:
            positions = actions
        elif self.action_format == ActionFormat.WAYPOINT_ANGLE:
            positions = actions[..., :2]
        elif self.action_format == ActionFormat.LINEAR_ANGULAR:
            # Roll out actions
            positions = torch.zeros_like(actions)
            heading = torch.zeros_like(actions[..., 0, 0])

            for i in range(1, actions.shape[-2]):
                vel = actions[..., i - 1, 0]
                angvel = actions[..., i - 1, 1]

                direction = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)
                positions[..., i, :] = positions[..., i - 1, :] + vel[..., None] * direction * self.dt
                heading = heading + angvel * self.dt
        else:
            raise ValueError(f"Unknown action format {self.action_format}")

        return positions

    def __getitem__(self, idx):
        # Sample a goal timestamp        
        ep_id = self.dataset_cache["episode_index"][idx].item()
        episode_length_remaining = self.episode_data_index["to"][ep_id] - idx
        goal_dist = np.random.randint(0, min(self.goal_horizon, episode_length_remaining))

        # Add the goal to the list of delta timestamps
        delta_timestamps = self.delta_timestamps or {k: [0.0] for k in item.keys()}
        delta_timestamps = delta_timestamps | {k: None for k in ["episode_index", "frame_index", "timestamp"]}
        delta_timestamps = {
            k: list(v) + [goal_dist * self.dt * self.action_spacing] if v is not None else None for k, v in delta_timestamps.items()
        }

        delta_timestamps["observation.filtered_position"] = delta_timestamps["observation.filtered_heading"] = delta_timestamps["observation.relative_position"] = [
            i * self.dt * self.action_spacing
            for i in range(self.goal_horizon)
        ] + [goal_dist * self.dt * self.action_spacing]

        item = load_frames_zarr(
            self.dataset_cache,
            idx,
            self.episode_data_index,
            delta_timestamps,
            self.tolerance_s,
        )
        
        flip_tf = random.random() > 0.5 
        image_obs = self._image_transforms(load_from_videos(
            {"observation.images.front": item["observation.images.front"][:-1]},
            ["observation.images.front"],
            self.videos_dir,
            self.tolerance_s,
            self.video_backend,
        )["observation.images.front"], flip_tf)
        image_goal = self._image_transforms(load_from_videos(
            {"observation.images.front": item["observation.images.front"][-1]},
            ["observation.images.front"],
            self.videos_dir,
            self.tolerance_s,
            self.video_backend,
        )["observation.images.front"], flip_tf)

        image_current, image_raw = self._image_transforms_depth(load_from_videos(
            {"observation.images.front": item["observation.images.front"][-2]},
            ["observation.images.front"],
            self.videos_dir,
            self.tolerance_s,
            self.video_backend,
        )["observation.images.front"], flip_tf)

        if self.sacson:
            delta_timestamps_sacson = {"pedestrian.filtered_position": [i * self.dt for i in range(-30, 5)]}
            item_pickle = load_pickle(
                self.dataset_cache,
                idx,
                self.episode_data_index,
                delta_timestamps_sacson,
            )      
            episode_length_prev = idx - self.episode_data_index["from"][ep_id]
            with open("/media/noriaki/Noriaki_Data/frodobots_dataset/ped_est/" + str(int(idx)) + ".pkl", 'rb') as file:
                data = pickle.load(file)               
            
            ped_dist = 1000000000.0
            ped_select = None
            for ip in data.keys():     
                dist = data[ip][0]**2 + data[ip][1]**2
                if dist < ped_dist:
                    ped_dist = dist
                    ped_select = ip
            
            ped_list = []            
            for id_ped in item_pickle:            
                with open("/media/noriaki/Noriaki_Data/frodobots_dataset/ped_est/" + str(int(id_ped)) + ".pkl", 'rb') as file:
                    data = pickle.load(file)    
                if ped_select in data.keys():                
                    ped_list.append(data[ped_select][0:2])   
                else:
                    ped_list.append([]) 
            
            head_trans = torch.from_numpy(self.dataset_cache["observation.filtered_heading"][item_pickle])            
            pos_trans = torch.from_numpy(self.dataset_cache["observation.filtered_position"][item_pickle]) 
            
            trans_cur_inv = torch.linalg.inv(trans_mat(pos_trans[30], head_trans[30]))
            
            ped_list_notrans = []
            ped_local_list = []
            robot_local_list = []            
            for istep in range(len(ped_list)):
                trans_fp = trans_mat(pos_trans[istep], head_trans[istep])            
                robot_local = torch.matmul(trans_cur_inv, trans_fp)
                if flip_tf:                
                    robot_local_list.append([robot_local[0,2], -robot_local[1,2]]) 
                else:
                    robot_local_list.append([robot_local[0,2], robot_local[1,2]]) 
                                    
                if ped_list[istep] != []:
                    trans_ped = torch.from_numpy(trans_mat([ped_list[istep][1], -ped_list[istep][0]], 0.0))
                    ped_local = torch.matmul(robot_local, trans_ped)
                    
                    if flip_tf:
                        ped_local_list.append([ped_local[0,2], -ped_local[1,2]])
                        ped_list_notrans.append([ped_list[istep][1], ped_list[istep][0]])     
                    else:
                        ped_local_list.append([ped_local[0,2], ped_local[1,2]])
                        ped_list_notrans.append([ped_list[istep][1], -ped_list[istep][0]])                                                        
                else:
                    ped_local_list.append([0.0, 0.0])
                    ped_list_notrans.append([0.0, 0.0])
            
            ped_local_list_raw = ped_local_list.copy()
            value_list = []
            for istep in range(len(ped_list)):
                if ped_local_list[istep] != [0.0, 0.0]:
                     value_list.append(istep)
            
            #interpolation                    
            for ii in range(len(value_list)-1):
                gap = float(value_list[ii+1] - value_list[ii])              
                delta_p = [(ped_local_list[value_list[ii+1]][0] - ped_local_list[value_list[ii]][0])/gap, (ped_local_list[value_list[ii+1]][1] - ped_local_list[value_list[ii]][1])/gap]
                for jj in range(value_list[ii]+1, value_list[ii+1], 1):
                    ped_local_list[jj] = [ped_local_list[value_list[ii]][0] + (jj - value_list[ii])*delta_p[0], ped_local_list[value_list[ii]][1] + (jj - value_list[ii])*delta_p[1]]

            ped_local_torch = torch.tensor(ped_local_list)
            for jp in range(5):
                for ip in range(len(ped_list)-2):   
                    if ped_local_torch[ip][0] != 0.0 and ped_local_torch[ip][1] != 0.0 and ped_local_torch[ip+1][0] != 0.0 and ped_local_torch[ip+1][1] != 0.0 and ped_local_torch[ip+2][0] != 0.0 and ped_local_torch[ip+2][1] != 0.0:
                        ped_local_torch[ip+1] = (ped_local_torch[ip] + ped_local_torch[ip+1] + ped_local_torch[ip+2])/3.0

            ped_local_slice = ped_local_torch[9:31:3]         
            ped_local_slice_raw = ped_local_list_raw[30:7:-3][::-1] #3fps  
            robot_local_slice = robot_local_list[30:7:-3][::-1] 
            ped_list_no_trans = ped_list_notrans[30:7:-3][::-1]
                               
        else:
            ped_list_no_trans = [0.0] #dummy
            ped_local_slice = [0.0] #dummy
            ped_local_slice_raw = [0.0] #dummy
            robot_local_slice = [0.0] #dummy
            
            
        unnorm_position = item["observation.filtered_position"][:-1]
        current_heading = item["observation.filtered_heading"][0]
        goal_heading = item["observation.filtered_heading"][-1]   
        heading = item["observation.filtered_heading"][:-1]        
        
        goal_pos_relative = to_local_coords(item["observation.filtered_position"][-1, None], unnorm_position[0], current_heading)[0]
        relative_mat = to_local_coords_yaw(item["observation.filtered_position"][-1, None], unnorm_position[0], current_heading, goal_heading)
        
        if flip_tf:
            goal_pos_relative[1] *= -1
            goal_heading *= -1
            relative_mat[0,1] *= -1
            relative_mat[1,0] *= -1
            relative_mat[1,2] *= -1
        
        if flip_tf:                  
            future_positions_unfiltered = to_local_coords(item["observation.relative_position"][:-1], unnorm_position[0], current_heading)                   
            direction = torch.stack([torch.cos(-heading), torch.sin(-heading)], dim=-1)
            action_steer = torch.clip(torch.from_numpy(np.diff(np.unwrap(-heading))), -1, 1) * 5           
            unnorm_position[:,1] *= -1            
            action_forward = torch.sum(torch.diff(unnorm_position, dim=0) * direction[:-1], dim=-1)
            action = torch.stack([action_forward[:self.action_horizon], action_steer[:self.action_horizon]], dim=-1) / self.dt / self.action_spacing               
            future_positions_unfiltered[:,1] *= -1   
        else:
            direction = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)
            action_steer = torch.clip(torch.from_numpy(np.diff(np.unwrap(heading))), -1, 1) * 5
            action_forward = torch.sum(torch.diff(unnorm_position, dim=0) * direction[:-1], dim=-1)
            action = torch.stack([action_forward[:self.action_horizon], action_steer[:self.action_horizon]], dim=-1) / self.dt / self.action_spacing        
            future_positions_unfiltered = to_local_coords(item["observation.relative_position"][:-1], unnorm_position[0], current_heading)

        which_dataset = 0

        image_flattened = einops.rearrange(image_obs, "... t c h w -> ... (t c) h w")
        return (
            torch.as_tensor(image_flattened, dtype=torch.float32),
            torch.as_tensor(image_goal, dtype=torch.float32),
            torch.as_tensor(image_current, dtype=torch.float32),            
            torch.as_tensor(action, dtype=torch.float32),
            torch.as_tensor(goal_dist/3.0, dtype=torch.int64),
            torch.as_tensor(goal_pos_relative, dtype=torch.float32),
            torch.as_tensor(relative_mat, dtype=torch.float32),  
            torch.as_tensor(goal_heading - current_heading, dtype=torch.float32),                        
            torch.as_tensor(which_dataset, dtype=torch.int64),
            torch.as_tensor(future_positions_unfiltered, dtype=torch.float32),
            torch.as_tensor(idx, dtype=torch.float32),
            torch.as_tensor(image_raw, dtype=torch.float32),     
            torch.as_tensor(ped_local_slice, dtype=torch.float32),         
            torch.as_tensor(ped_local_slice_raw, dtype=torch.float32),        
            torch.as_tensor(ped_list_no_trans, dtype=torch.float32),                            
            torch.as_tensor(robot_local_slice, dtype=torch.float32),                                   
        )  

    def get_sampler(self, base_rate: float = 0.1):
        """
        Create a sampler that samples dataset elements proportionally to the sum of squared future turning actions (+ base_rate).

        A sample that drives straight will be weighted by base_rate, while a sample that is constantly turning at max speed will be weighted by 1.
        """
        indices = torch.arange(len(self))
        to_indices = self.episode_data_index["to"] - 1
        to_indices = to_indices[self.dataset_cache["episode_index"]]

        target_indices = indices[:, None] + torch.arange(self.action_horizon) * self.action_spacing
        target_next_indices = target_indices + 1
        target_indices.clip_(indices[:, None], to_indices[:, None])
        target_next_indices.clip_(indices[:, None], to_indices[:, None])

        headings = torch.tensor(self.dataset_cache["observation.filtered_heading"])
        heading_diff = (headings[target_indices] - headings[target_next_indices]).clip_(-0.2, 0.2).abs_().sum(dim=-1)
        future_steer = torch.clip(heading_diff, -1, 1)
        weights = base_rate + (1 - base_rate) * future_steer ** 2

        return torch.utils.data.WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

class ViNTLeRobotDataset_IL2_gps(LeRobotDataset):
    def __init__(
        self,
        repo_id: str,
        video: str,
        root: Path | None,
        split: str = "train",
        action_format: ActionFormat | str = ActionFormat.WAYPOINT,
        action_horizon: int = 8,
        action_spacing: int = 1,
        goal_horizon: int = 20,
        goal_horizon2: int = 20,        
        context_size: int = 5,
        context_spacing: int = 1,
        dataset_framerate: int = 10,
        image_size: Tuple[int, int] = (120, 160),
        image_transforms: Callable | None = None,
        sacson: bool = False
    ):
        """
        Main ViNT dataset class
        """
        if isinstance(action_format, str):
            action_format = ActionFormat.from_str(action_format)
        self.action_format = action_format

        if action_format == ActionFormat.WAYPOINT:
            self.num_action_params = 2
        elif action_format == ActionFormat.WAYPOINT_ANGLE:
            self.num_action_params = 3
        elif action_format == ActionFormat.LINEAR_ANGULAR:
            self.num_action_params = 2

        self.dt = 1 / dataset_framerate
        self.action_spacing = action_spacing
        self.action_horizon = action_horizon
        self.goal_horizon = goal_horizon
        self.goal_horizon2 = goal_horizon2        
        self.context_size = context_size
        self.context_spacing = context_spacing
        self.image_size = image_size
        self.sacson = sacson

        print("root_init", root)
        super().__init__(
            repo_id=repo_id,
            video=video,
            root=root,
            split=split,
            image_transforms=image_transforms,
            delta_timestamps={
                "observation.filtered_position": [0.0],
                "observation.relative_position": [0.0],
                "observation.filtered_heading": [0.0],
                "observation.images.front": [i * context_spacing * self.dt for i in range(-context_size, 1)],
                "action": [i * action_spacing * self.dt for i in range(action_horizon)],                
            },
        )

        # Build a cache of episode data indices
        self.dataset_cache = zarr.load(Path(root) / "frodobots_dataset" / "dataset_cache.zarr")
        self.dataset_cache = {
            k: np.asarray(v) for k, v in self.dataset_cache.items()
        }
        self.min_action_distance = 3
        self.max_action_distance = 20
        
    def _image_transforms(self, img: torch.Tensor, flip) -> torch.Tensor:
        """
        Args:
            img (torch.Tensor): image tensor
        Returns:
            torch.Tensor: transformed image
        """
        if self.image_transforms is not None:
            img = self.image_transforms(img)

        original_height, original_width = img.shape[-2:]
        target_aspect = 4 / 3
        img = TF.resize(img, self.image_size)
        
        if flip:
            img = torch.flip(img, dims=(-1,))
        
        return img

    def _image_transforms_depth(self, img: torch.Tensor, flip) -> torch.Tensor:
        """
        Args:
            img (torch.Tensor): image tensor
        Returns:
            torch.Tensor: transformed image
        """

        img_rsize = TF.resize(img, (128, 416)) #               
        if flip:
            img_rsize = torch.flip(img_rsize, dims=(-1,))
            img = torch.flip(img, dims=(-1,))                   
        return img_rsize, img

    def viz_rollout(self, actions: torch.Tensor) -> torch.Tensor:
        if self.action_format == ActionFormat.WAYPOINT:
            positions = actions
        elif self.action_format == ActionFormat.WAYPOINT_ANGLE:
            positions = actions[..., :2]
        elif self.action_format == ActionFormat.LINEAR_ANGULAR:
            # Roll out actionsViNTLeRobotDataset_IL2
            positions = torch.zeros_like(actions)
            heading = torch.zeros_like(actions[..., 0, 0])

            for i in range(1, actions.shape[-2]):
                vel = actions[..., i - 1, 0]
                angvel = actions[..., i - 1, 1]

                direction = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)
                positions[..., i, :] = positions[..., i - 1, :] + vel[..., None] * direction * self.dt
                heading = heading + angvel * self.dt
        else:
            raise ValueError(f"Unknown action format {self.action_format}")

        return positions

    def __getitem__(self, idx):
        # Sample a goal timestamp        
        ep_id = self.dataset_cache["episode_index"][idx].item()
        episode_length_remaining = self.episode_data_index["to"][ep_id] - idx
        goal_dist = np.random.randint(0, min(self.goal_horizon, episode_length_remaining))
        goal_dist2 = min(8, episode_length_remaining)
        goal_dist_gps = np.random.randint(0, min(self.goal_horizon2, episode_length_remaining))    

        # Add the goal to the list of delta timestamps
        delta_timestamps = self.delta_timestamps or {k: [0.0] for k in item.keys()}
        delta_timestamps = delta_timestamps | {k: None for k in ["episode_index", "frame_index", "timestamp"]}
        delta_timestamps = {
            k: list(v) + [goal_dist * self.dt * self.action_spacing] + [goal_dist2 * self.dt * self.action_spacing] + [goal_dist_gps * self.dt * self.action_spacing] if v is not None else None for k, v in delta_timestamps.items()
        }

        control_horizon = 8+1
        delta_timestamps["observation.filtered_position"] = delta_timestamps["observation.filtered_heading"] = delta_timestamps["observation.relative_position"] = [
            i * self.dt * self.action_spacing
            for i in range(control_horizon)
        ] + [goal_dist * self.dt * self.action_spacing] + [goal_dist_gps * self.dt * self.action_spacing]
        item = load_frames_zarr(
            self.dataset_cache,
            idx,
            self.episode_data_index,
            delta_timestamps,
            self.tolerance_s,
        )
        
        flip_tf = random.random() > 0.5     
        image_obs = self._image_transforms(load_from_videos(
            {"observation.images.front": item["observation.images.front"][:-3]},
            ["observation.images.front"],
            self.videos_dir,
            self.tolerance_s,
            self.video_backend,
        )["observation.images.front"], flip_tf)
        
        image_goal2 = self._image_transforms(load_from_videos(
            {"observation.images.front": item["observation.images.front"][-2]},
            ["observation.images.front"],
            self.videos_dir,
            self.tolerance_s,
            self.video_backend,
        )["observation.images.front"], flip_tf) #for inverse dynamics model
                
        image_goal = self._image_transforms(load_from_videos(
            {"observation.images.front": item["observation.images.front"][-3]},
            ["observation.images.front"],
            self.videos_dir,
            self.tolerance_s,
            self.video_backend,
        )["observation.images.front"], flip_tf)        
        
        image_current, image_raw = self._image_transforms_depth(load_from_videos(
            {"observation.images.front": item["observation.images.front"][-4]},
            ["observation.images.front"],
            self.videos_dir,
            self.tolerance_s,
            self.video_backend,
        )["observation.images.front"], flip_tf)
        
        if self.sacson:
            delta_timestamps_sacson = {"pedestrian.filtered_position": [i * self.dt for i in range(-30, 5)]}
            item_pickle = load_pickle(
                self.dataset_cache,
                idx,
                self.episode_data_index,
                delta_timestamps_sacson,
            )      
            episode_length_prev = idx - self.episode_data_index["from"][ep_id]
            with open("/media/noriaki/Noriaki_Data/frodobots_dataset/ped_est/" + str(int(idx)) + ".pkl", 'rb') as file:
                data = pickle.load(file)                          
                
            ped_dist = 1000000000.0
            ped_select = None
            for ip in data.keys():     
                dist = data[ip][0]**2 + data[ip][1]**2
                if dist < ped_dist:
                    ped_dist = dist
                    ped_select = ip
            
            ped_list = []            
            for id_ped in item_pickle:            
                with open("/media/noriaki/Noriaki_Data/frodobots_dataset/ped_est/" + str(int(id_ped)) + ".pkl", 'rb') as file:
                    data = pickle.load(file)     
                    
                if ped_select in data.keys():                
                    ped_list.append(data[ped_select][0:2])   
                else:
                    ped_list.append([])    
            
            head_trans = torch.from_numpy(self.dataset_cache["observation.filtered_heading"][item_pickle])            
            pos_trans = torch.from_numpy(self.dataset_cache["observation.filtered_position"][item_pickle]) 
            
            trans_cur_inv = torch.linalg.inv(trans_mat(pos_trans[30], head_trans[30]))
            
            ped_list_notrans = []
            ped_local_list = []
            robot_local_list = []            
            for istep in range(len(ped_list)):
                trans_fp = trans_mat(pos_trans[istep], head_trans[istep])            
                robot_local = torch.matmul(trans_cur_inv, trans_fp)
                robot_local_list.append([robot_local[0,2], robot_local[1,2]]) 
                if ped_list[istep] != []:
                    trans_ped = torch.from_numpy(trans_mat([ped_list[istep][1], -ped_list[istep][0]], 0.0))
                    ped_local = torch.matmul(robot_local, trans_ped)
                    ped_local_list.append([ped_local[0,2], ped_local[1,2]])
                    ped_list_notrans.append([ped_list[istep][1], -ped_list[istep][0]])                                       
                else:
                    ped_local_list.append([0.0, 0.0])
                    ped_list_notrans.append([0.0, 0.0])
            
            ped_local_list_raw = ped_local_list.copy()
            value_list = []
            for istep in range(len(ped_list)):
                if ped_local_list[istep] != [0.0, 0.0]:
                     value_list.append(istep)
            
            #interpolation                    
            for ii in range(len(value_list)-1):
                gap = float(value_list[ii+1] - value_list[ii])              
                delta_p = [(ped_local_list[value_list[ii+1]][0] - ped_local_list[value_list[ii]][0])/gap, (ped_local_list[value_list[ii+1]][1] - ped_local_list[value_list[ii]][1])/gap]
                for jj in range(value_list[ii]+1, value_list[ii+1], 1):
                    ped_local_list[jj] = [ped_local_list[value_list[ii]][0] + (jj - value_list[ii])*delta_p[0], ped_local_list[value_list[ii]][1] + (jj - value_list[ii])*delta_p[1]]
       
            ped_local_torch = torch.tensor(ped_local_list)
            for jp in range(5):
                for ip in range(len(ped_list)-2):   
                    if ped_local_torch[ip][0] != 0.0 and ped_local_torch[ip][1] != 0.0 and ped_local_torch[ip+1][0] != 0.0 and ped_local_torch[ip+1][1] != 0.0 and ped_local_torch[ip+2][0] != 0.0 and ped_local_torch[ip+2][1] != 0.0:
                        ped_local_torch[ip+1] = (ped_local_torch[ip] + ped_local_torch[ip+1] + ped_local_torch[ip+2])/3.0

            ped_local_slice = ped_local_torch[9:31:3]
            ped_local_slice_raw = ped_local_list_raw[30:7:-3][::-1] #3fps  
            robot_local_slice = robot_local_list[30:7:-3][::-1] 
            ped_list_no_trans = ped_list_notrans[30:7:-3][::-1]
                               
        else:
            ped_list_no_trans = [0.0] #dummy
            ped_local_slice = [0.0] #dummy
            ped_local_slice_raw = [0.0] #dummy
            robot_local_slice = [0.0] #dummy
         
        unnorm_position = item["observation.filtered_position"][:-1]
        current_heading = item["observation.filtered_heading"][0]
        goal_heading = item["observation.filtered_heading"][-1]
        
        if flip_tf:
            goal_pos_relative = to_local_coords(item["observation.filtered_position"][-1, None], unnorm_position[0], current_heading)[0]        
            relative_mat = to_local_coords_yaw(item["observation.filtered_position"][-1, None], unnorm_position[0], current_heading, goal_heading)       
            goal_pos_relative[1] *= -1                
            relative_mat[0,1] *= -1
            relative_mat[1,0] *= -1                                    
        else:
            goal_pos_relative = to_local_coords(item["observation.filtered_position"][-1, None], unnorm_position[0], current_heading)[0]
            relative_mat = to_local_coords_yaw(item["observation.filtered_position"][-1, None], unnorm_position[0], current_heading, goal_heading)        
         
        action_IL = []
        metric_waypoint_spacing = 0.25 #normalization        
        goal_pos_relative = goal_pos_relative/metric_waypoint_spacing #normalization        
        for i_traj in range(control_horizon-1):
            traj_relative_mat = to_local_coords_yaw(item["observation.filtered_position"][i_traj + 1, None], unnorm_position[0], current_heading, item["observation.filtered_heading"][i_traj + 1])    
            
            if flip_tf:
                action_IL.append([traj_relative_mat[0,2]/metric_waypoint_spacing, -traj_relative_mat[1,2]/metric_waypoint_spacing, traj_relative_mat[1,1], -traj_relative_mat[1,0]])
            else:
                action_IL.append([traj_relative_mat[0,2]/metric_waypoint_spacing, traj_relative_mat[1,2]/metric_waypoint_spacing, traj_relative_mat[1,1], traj_relative_mat[1,0]])
                            
        action_IL = torch.tensor(action_IL)        
        heading = item["observation.filtered_heading"][:-1]
        direction = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)
        
        action_steer = torch.clip(torch.from_numpy(np.diff(np.unwrap(heading))), -1, 1) * 5
        action_forward = torch.sum(torch.diff(unnorm_position, dim=0) * direction[:-1], dim=-1)
        action = torch.stack([action_forward[:self.action_horizon], action_steer[:self.action_horizon]], dim=-1) / self.dt / self.action_spacing

        future_positions_unfiltered = to_local_coords(item["observation.relative_position"][:-1], unnorm_position[0], current_heading)
        which_dataset = 0
        image_flattened = einops.rearrange(image_obs, "... t c h w -> ... (t c) h w")

        if goal_dist == 0:
            goal_is_negative = True
        else:
            goal_is_negative = False
        #[TODO] in GNM dataset, we set goal_dist == 20 when goal_dist is zero (current frame == goal frame) and we give random goal image. But we do in training code. We need to fix this not to confuse the users.
        
        action_mask = (
            (goal_dist < self.max_action_distance) and
            (goal_dist > self.min_action_distance) and
            (not goal_is_negative)
        )
        return (
            torch.as_tensor(image_flattened, dtype=torch.float32),
            torch.as_tensor(image_goal, dtype=torch.float32),
            torch.as_tensor(image_goal2, dtype=torch.float32),            
            torch.as_tensor(image_current, dtype=torch.float32),            
            torch.as_tensor(action_IL, dtype=torch.float32),
            torch.as_tensor(goal_dist, dtype=torch.int64),
            torch.as_tensor(goal_pos_relative, dtype=torch.float32),
            torch.as_tensor(relative_mat, dtype=torch.float32),  
            torch.as_tensor(goal_heading - current_heading, dtype=torch.float32),                        
            torch.as_tensor(which_dataset, dtype=torch.int64),
            torch.as_tensor(future_positions_unfiltered, dtype=torch.float32),
            torch.as_tensor(idx, dtype=torch.float32),
            torch.as_tensor(action_mask, dtype=torch.float32),           
            torch.as_tensor(ped_local_slice, dtype=torch.float32),         
            torch.as_tensor(ped_local_slice_raw, dtype=torch.float32),        
            torch.as_tensor(ped_list_no_trans, dtype=torch.float32),
            torch.as_tensor(robot_local_slice, dtype=torch.float32),                                   
        )  

    def get_sampler(self, base_rate: float = 0.1):
        """
        Create a sampler that samples dataset elements proportionally to the sum of squared future turning actions (+ base_rate).

        A sample that drives straight will be weighted by base_rate, while a sample that is constantly turning at max speed will be weighted by 1.
        """
        indices = torch.arange(len(self))
        to_indices = self.episode_data_index["to"] - 1
        to_indices = to_indices[self.dataset_cache["episode_index"]]

        target_indices = indices[:, None] + torch.arange(self.action_horizon) * self.action_spacing
        target_next_indices = target_indices + 1
        target_indices.clip_(indices[:, None], to_indices[:, None])
        target_next_indices.clip_(indices[:, None], to_indices[:, None])

        headings = torch.tensor(self.dataset_cache["observation.filtered_heading"])
        heading_diff = (headings[target_indices] - headings[target_next_indices]).clip_(-0.2, 0.2).abs_().sum(dim=-1)
        future_steer = torch.clip(heading_diff, -1, 1)
        weights = base_rate + (1 - base_rate) * future_steer ** 2

        return torch.utils.data.WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


class ViNTLeRobotDataset_IL2(LeRobotDataset):
    def __init__(
        self,
        repo_id: str,
        video: str,
        root: Path | None,
        split: str = "train",
        action_format: ActionFormat | str = ActionFormat.WAYPOINT,
        action_horizon: int = 8,
        action_spacing: int = 1,
        goal_horizon: int = 20,
        context_size: int = 5,
        context_spacing: int = 1,
        dataset_framerate: int = 10,
        image_size: Tuple[int, int] = (120, 160),
        image_transforms: Callable | None = None,
        sacson: bool = False
    ):
        """
        Main ViNT dataset class
        """
        if isinstance(action_format, str):
            action_format = ActionFormat.from_str(action_format)
        self.action_format = action_format

        if action_format == ActionFormat.WAYPOINT:
            self.num_action_params = 2
        elif action_format == ActionFormat.WAYPOINT_ANGLE:
            self.num_action_params = 3
        elif action_format == ActionFormat.LINEAR_ANGULAR:
            self.num_action_params = 2

        self.dt = 1 / dataset_framerate
        self.action_spacing = action_spacing
        self.action_horizon = action_horizon
        self.goal_horizon = goal_horizon
        self.context_size = context_size
        self.context_spacing = context_spacing
        self.image_size = image_size
        self.sacson = sacson

        print("root_init", root)
        super().__init__(
            repo_id=repo_id,
            video=video,
            root=root,
            split=split,
            image_transforms=image_transforms,
            delta_timestamps={
                "observation.filtered_position": [0.0],
                "observation.relative_position": [0.0],
                "observation.filtered_heading": [0.0],
                "observation.images.front": [i * context_spacing * self.dt for i in range(-context_size, 1)],
                "action": [i * action_spacing * self.dt for i in range(action_horizon)],                
            },
        )

        # Build a cache of episode data indices
        self.dataset_cache = zarr.load(Path(root) / "frodobots_dataset" / "dataset_cache.zarr")        
        self.dataset_cache = {
            k: np.asarray(v) for k, v in self.dataset_cache.items()
        }

        self.min_action_distance = 3
        self.max_action_distance = 20
        
    def _image_transforms(self, img: torch.Tensor, flip) -> torch.Tensor:
        """
        Args:
            img (torch.Tensor): image tensor
        Returns:
            torch.Tensor: transformed image
        """
        if self.image_transforms is not None:
            img = self.image_transforms(img)

        original_height, original_width = img.shape[-2:]
        target_aspect = 4 / 3
        img = TF.resize(img, self.image_size)
        
        if flip:
            img = torch.flip(img, dims=(-1,))
        
        return img

    def _image_transforms_depth(self, img: torch.Tensor, flip) -> torch.Tensor:
        """
        Args:
            img (torch.Tensor): image tensor
        Returns:
            torch.Tensor: transformed image
        """

        img_rsize = TF.resize(img, (128, 416)) #
        if flip:
            img_rsize = torch.flip(img_rsize, dims=(-1,))
            img = torch.flip(img, dims=(-1,))
                                            
        return img_rsize, img

    def viz_rollout(self, actions: torch.Tensor) -> torch.Tensor:
        if self.action_format == ActionFormat.WAYPOINT:
            positions = actions
        elif self.action_format == ActionFormat.WAYPOINT_ANGLE:
            positions = actions[..., :2]
        elif self.action_format == ActionFormat.LINEAR_ANGULAR:
            # Roll out actions
            positions = torch.zeros_like(actions)
            heading = torch.zeros_like(actions[..., 0, 0])

            for i in range(1, actions.shape[-2]):
                vel = actions[..., i - 1, 0]
                angvel = actions[..., i - 1, 1]

                direction = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)
                positions[..., i, :] = positions[..., i - 1, :] + vel[..., None] * direction * self.dt
                heading = heading + angvel * self.dt
        else:
            raise ValueError(f"Unknown action format {self.action_format}")

        return positions

    def __getitem__(self, idx):
        # Sample a goal timestamp        
        ep_id = self.dataset_cache["episode_index"][idx].item()
        episode_length_remaining = self.episode_data_index["to"][ep_id] - idx
        goal_dist = np.random.randint(0, min(self.goal_horizon, episode_length_remaining))
        goal_dist2 = min(8, episode_length_remaining)

        # Add the goal to the list of delta timestamps
        delta_timestamps = self.delta_timestamps or {k: [0.0] for k in item.keys()}
        delta_timestamps = delta_timestamps | {k: None for k in ["episode_index", "frame_index", "timestamp"]}
        delta_timestamps = {
            k: list(v) + [goal_dist * self.dt * self.action_spacing]  + [goal_dist2 * self.dt * self.action_spacing] if v is not None else None for k, v in delta_timestamps.items()
        }

        control_horizon = 8+1
        delta_timestamps["observation.filtered_position"] = delta_timestamps["observation.filtered_heading"] = delta_timestamps["observation.relative_position"] = [
            i * self.dt * self.action_spacing
            for i in range(control_horizon)
        ] + [goal_dist * self.dt * self.action_spacing]
        
        item = load_frames_zarr(
            self.dataset_cache,
            idx,
            self.episode_data_index,
            delta_timestamps,
            self.tolerance_s,
        )
        
        flip_tf = random.random() > 0.5    
        image_obs = self._image_transforms(load_from_videos(
            {"observation.images.front": item["observation.images.front"][:-2]},
            ["observation.images.front"],
            self.videos_dir,
            self.tolerance_s,
            self.video_backend,
        )["observation.images.front"], flip_tf)
        
        image_goal2 = self._image_transforms(load_from_videos(
            {"observation.images.front": item["observation.images.front"][-1]},
            ["observation.images.front"],
            self.videos_dir,
            self.tolerance_s,
            self.video_backend,
        )["observation.images.front"], flip_tf) #for inverse dynamics model
                
        image_goal = self._image_transforms(load_from_videos(
            {"observation.images.front": item["observation.images.front"][-2]},
            ["observation.images.front"],
            self.videos_dir,
            self.tolerance_s,
            self.video_backend,
        )["observation.images.front"], flip_tf)
           
        image_current, image_raw = self._image_transforms_depth(load_from_videos(
            {"observation.images.front": item["observation.images.front"][-3]},
            ["observation.images.front"],
            self.videos_dir,
            self.tolerance_s,
            self.video_backend,
        )["observation.images.front"], flip_tf)
        
        
        if self.sacson:
            delta_timestamps_sacson = {"pedestrian.filtered_position": [i * self.dt for i in range(-30, 5)]}
            item_pickle = load_pickle(
                self.dataset_cache,
                idx,
                self.episode_data_index,
                delta_timestamps_sacson,
            )      
            episode_length_prev = idx - self.episode_data_index["from"][ep_id]
            with open("/media/noriaki/Noriaki_Data/frodobots_dataset/ped_est/" + str(int(idx)) + ".pkl", 'rb') as file:
                data = pickle.load(file)                           
                
            ped_dist = 1000000000.0
            ped_select = None
            for ip in data.keys():     
                dist = data[ip][0]**2 + data[ip][1]**2
                if dist < ped_dist:
                    ped_dist = dist
                    ped_select = ip
            
            ped_list = []            
            for id_ped in item_pickle:            
                with open("/media/noriaki/Noriaki_Data/frodobots_dataset/ped_est/" + str(int(id_ped)) + ".pkl", 'rb') as file:
                    data = pickle.load(file)    
                    
                if ped_select in data.keys():                
                    ped_list.append(data[ped_select][0:2])   
                else:
                    ped_list.append([])
            
            head_trans = torch.from_numpy(self.dataset_cache["observation.filtered_heading"][item_pickle])            
            pos_trans = torch.from_numpy(self.dataset_cache["observation.filtered_position"][item_pickle]) 
            
            trans_cur_inv = torch.linalg.inv(trans_mat(pos_trans[30], head_trans[30]))
            
            ped_list_notrans = []
            ped_local_list = []
            robot_local_list = []            
            for istep in range(len(ped_list)):
                trans_fp = trans_mat(pos_trans[istep], head_trans[istep])            
                robot_local = torch.matmul(trans_cur_inv, trans_fp)
                robot_local_list.append([robot_local[0,2], robot_local[1,2]]) 
                if ped_list[istep] != []:
                    trans_ped = torch.from_numpy(trans_mat([ped_list[istep][1], -ped_list[istep][0]], 0.0))
                    ped_local = torch.matmul(robot_local, trans_ped)
                    ped_local_list.append([ped_local[0,2], ped_local[1,2]])
                    ped_list_notrans.append([ped_list[istep][1], -ped_list[istep][0]])                                       
                else:
                    ped_local_list.append([0.0, 0.0])
                    ped_list_notrans.append([0.0, 0.0])
            
            ped_local_list_raw = ped_local_list.copy()
            value_list = []
            for istep in range(len(ped_list)):
                if ped_local_list[istep] != [0.0, 0.0]:
                     value_list.append(istep)
            
            #interpolation                    
            for ii in range(len(value_list)-1):
                gap = float(value_list[ii+1] - value_list[ii])              
                delta_p = [(ped_local_list[value_list[ii+1]][0] - ped_local_list[value_list[ii]][0])/gap, (ped_local_list[value_list[ii+1]][1] - ped_local_list[value_list[ii]][1])/gap]
                for jj in range(value_list[ii]+1, value_list[ii+1], 1):
                    ped_local_list[jj] = [ped_local_list[value_list[ii]][0] + (jj - value_list[ii])*delta_p[0], ped_local_list[value_list[ii]][1] + (jj - value_list[ii])*delta_p[1]]
            
            ped_local_torch = torch.tensor(ped_local_list)
            for jp in range(5):
                for ip in range(len(ped_list)-2):   
                    if ped_local_torch[ip][0] != 0.0 and ped_local_torch[ip][1] != 0.0 and ped_local_torch[ip+1][0] != 0.0 and ped_local_torch[ip+1][1] != 0.0 and ped_local_torch[ip+2][0] != 0.0 and ped_local_torch[ip+2][1] != 0.0:
                        ped_local_torch[ip+1] = (ped_local_torch[ip] + ped_local_torch[ip+1] + ped_local_torch[ip+2])/3.0

            ped_local_slice = ped_local_torch[9:31:3]         
            ped_local_slice_raw = ped_local_list_raw[30:7:-3][::-1] #3fps  
            robot_local_slice = robot_local_list[30:7:-3][::-1] 
            ped_list_no_trans = ped_list_notrans[30:7:-3][::-1]
                               
        else:
            ped_list_no_trans = [0.0] #dummy
            ped_local_slice = [0.0] #dummy
            ped_local_slice_raw = [0.0] #dummy
            robot_local_slice = [0.0] #dummy
          
        unnorm_position = item["observation.filtered_position"][:-1]
        current_heading = item["observation.filtered_heading"][0]
        goal_heading = item["observation.filtered_heading"][-1]
        
        goal_pos_relative = to_local_coords(item["observation.filtered_position"][-1, None], unnorm_position[0], current_heading)[0]
        relative_mat = to_local_coords_yaw(item["observation.filtered_position"][-1, None], unnorm_position[0], current_heading, goal_heading)
        
        action_IL = []
        metric_waypoint_spacing = 0.25 #normalization        
        for i_traj in range(control_horizon-1):
            traj_relative_mat = to_local_coords_yaw(item["observation.filtered_position"][i_traj + 1, None], unnorm_position[0], current_heading, item["observation.filtered_heading"][i_traj + 1])    
            
            if flip_tf:
                action_IL.append([traj_relative_mat[0,2]/metric_waypoint_spacing, -traj_relative_mat[1,2]/metric_waypoint_spacing, traj_relative_mat[1,1], -traj_relative_mat[1,0]])
            else:
                action_IL.append([traj_relative_mat[0,2]/metric_waypoint_spacing, traj_relative_mat[1,2]/metric_waypoint_spacing, traj_relative_mat[1,1], traj_relative_mat[1,0]])
                            
        action_IL = torch.tensor(action_IL)        
        heading = item["observation.filtered_heading"][:-1]
        direction = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)
        
        action_steer = torch.clip(torch.from_numpy(np.diff(np.unwrap(heading))), -1, 1) * 5
        action_forward = torch.sum(torch.diff(unnorm_position, dim=0) * direction[:-1], dim=-1)
        action = torch.stack([action_forward[:self.action_horizon], action_steer[:self.action_horizon]], dim=-1) / self.dt / self.action_spacing
        
        future_positions_unfiltered = to_local_coords(item["observation.relative_position"][:-1], unnorm_position[0], current_heading)
        which_dataset = 0

        image_flattened = einops.rearrange(image_obs, "... t c h w -> ... (t c) h w")

        if goal_dist == 0:
            goal_is_negative = True
        else:
            goal_is_negative = False
        #[TODO] in GNM dataset, we set goal_dist == 20 when goal_dist is zero (current frame == goal frame) and we give random goal image. But we do in training code. We need to fix this not to confuse the users.
        
        
        action_mask = (
            (goal_dist < self.max_action_distance) and
            (goal_dist > self.min_action_distance) and
            (not goal_is_negative)
        )
        return (
            torch.as_tensor(image_flattened, dtype=torch.float32),
            torch.as_tensor(image_goal, dtype=torch.float32),
            torch.as_tensor(image_goal2, dtype=torch.float32),            
            torch.as_tensor(image_current, dtype=torch.float32),            
            torch.as_tensor(action_IL, dtype=torch.float32),
            torch.as_tensor(goal_dist, dtype=torch.int64),
            torch.as_tensor(goal_pos_relative, dtype=torch.float32),
            torch.as_tensor(relative_mat, dtype=torch.float32),  
            torch.as_tensor(goal_heading - current_heading, dtype=torch.float32),                        
            torch.as_tensor(which_dataset, dtype=torch.int64),
            torch.as_tensor(future_positions_unfiltered, dtype=torch.float32),
            torch.as_tensor(idx, dtype=torch.float32),
            torch.as_tensor(action_mask, dtype=torch.float32),            
            torch.as_tensor(ped_local_slice, dtype=torch.float32),         
            torch.as_tensor(ped_local_slice_raw, dtype=torch.float32),        
            torch.as_tensor(ped_list_no_trans, dtype=torch.float32),
            torch.as_tensor(robot_local_slice, dtype=torch.float32),                                   
        )  

    def get_sampler(self, base_rate: float = 0.1):
        """
        Create a sampler that samples dataset elements proportionally to the sum of squared future turning actions (+ base_rate).

        A sample that drives straight will be weighted by base_rate, while a sample that is constantly turning at max speed will be weighted by 1.
        """
        indices = torch.arange(len(self))
        to_indices = self.episode_data_index["to"] - 1
        to_indices = to_indices[self.dataset_cache["episode_index"]]

        target_indices = indices[:, None] + torch.arange(self.action_horizon) * self.action_spacing
        target_next_indices = target_indices + 1
        target_indices.clip_(indices[:, None], to_indices[:, None])
        target_next_indices.clip_(indices[:, None], to_indices[:, None])

        headings = torch.tensor(self.dataset_cache["observation.filtered_heading"])
        heading_diff = (headings[target_indices] - headings[target_next_indices]).clip_(-0.2, 0.2).abs_().sum(dim=-1)
        future_steer = torch.clip(heading_diff, -1, 1)
        weights = base_rate + (1 - base_rate) * future_steer ** 2
        return torch.utils.data.WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)



class EpisodeSampler_IL(torch.utils.data.Sampler):
    def __init__(self, dataset: ViNTLeRobotDataset, episode_index_from: int, episode_index_to: int, goal_horizon: int, data_split_type: str):
        self.dataset = dataset
        self.goal_horizon = goal_horizon
        from_idx = dataset.episode_data_index["from"][episode_index_from].item()
        to_idx = dataset.episode_data_index["to"][episode_index_to].item()
        self.frame_ids_range = range(from_idx, to_idx)
        print("from_idx", from_idx, "to_idx", to_idx)  

        if data_split_type == "train":
            with open('./vint_train/data/sampler/train_yaw_small.pkl', 'rb') as file:
                data = pickle.load(file)
            with open('./vint_train/data/sampler/train_ped_fix.pkl', 'rb') as file:
                data_ped = pickle.load(file)                
        elif data_split_type == "test":
            with open('./vint_train/data/sampler/test_yaw_small.pkl', 'rb') as file:
                data = pickle.load(file)
            with open('./vint_train/data/sampler/test_ped_fix.pkl', 'rb') as file:
                data_ped = pickle.load(file)         
     
        self.yaw_list = data[1]
        self.ped_list = data_ped[1]        
        self.init_idx = data[0][0]                        
                                      
    def __iter__(self) -> Iterator:   
        indices_new = []
        yawangle_list = []
        
        for idx in tqdm(self.frame_ids_range):
            
            thres_rate = random.random()
            if self.yaw_list[idx-self.init_idx] % (2*3.14) > 3.14:
                ang_yaw = self.yaw_list[idx-self.init_idx] % (2*3.14) - 2.0*3.14
            else:
                ang_yaw = self.yaw_list[idx-self.init_idx] % (2*3.14)   
            
            while abs(ang_yaw) > 2.0:
                idx = random.choice(self.frame_ids_range)   
                if self.yaw_list[idx-self.init_idx] % (2*3.14) > 3.14:
                    ang_yaw = self.yaw_list[idx-self.init_idx] % (2*3.14) - 2.0*3.14
                else:
                    ang_yaw = self.yaw_list[idx-self.init_idx] % (2*3.14)   
            
            if thres_rate < 0.8:
                while not ((abs(ang_yaw) > 0.4 or self.ped_list[idx-self.init_idx] != 10000.0) and abs(ang_yaw) < 2.0):                  
                    idx = random.choice(self.frame_ids_range)            
                    if self.yaw_list[idx-self.init_idx] % (2*3.14) > 3.14:
                        ang_yaw = self.yaw_list[idx-self.init_idx] % (2*3.14) - 2.0*3.14
                    else:
                        ang_yaw = self.yaw_list[idx-self.init_idx] % (2*3.14)             
            
            indices_new.append(idx)           

        indices_new_random = random.sample(indices_new, len(indices_new)) 
        return iter(indices_new_random)

    def __len__(self) -> int:
        return len(self.frame_ids_range)   
            
