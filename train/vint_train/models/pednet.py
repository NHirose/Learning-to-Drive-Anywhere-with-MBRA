#!/usr/bin/env python3

import torch
from torch import nn
import torch.nn.functional as F

class PedNet(nn.Module):
    def __init__(self, in_step_size, out_step_size):
        super(PedNet, self).__init__()

        self.model_1 = nn.Sequential(
            nn.Linear(2 * 2 * in_step_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),            
        )
        self.model_2 = nn.Sequential(
            nn.Linear(32 + 2*in_step_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 2 * in_step_size),
            nn.Tanh(),
        )

    def forward(self, ped_traj_p, robot_traj_p, robot_traj_f):
        x1 = self.model_1(torch.cat((ped_traj_p, robot_traj_p), axis=1))
        x2 = torch.cat((x1, robot_traj_f), axis=1)
        y = 0.333*1.5*self.model_2(x2) #0.333 = 3fps 1.5 = 1.5 m/s
        return y
        
def unddp_state_dict(state_dict):
    """convert DDP-trained module's state dict to raw module's.

    Nothing is done if state_dict isn't DDP-trained.

    """

    if not all([s.startswith("module.") for s in state_dict.keys()]):
        return state_dict

    return OrderedDict((k[7:], v) for k, v in state_dict.items())

