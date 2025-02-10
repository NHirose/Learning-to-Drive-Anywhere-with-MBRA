# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from .layers import *

class DepthDecoder_camera_ada4(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_bins=32, num_output_channels=1, use_skips=True):
        super(DepthDecoder_camera_ada4, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        self.num_bins = num_bins
        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.convs[("conv1_c")] = nn.Conv2d(512, 512, 3, 2, 1)
        self.convs[("conv2_c")] = nn.Conv2d(512, 512, 3, 2, 1)
        self.convs[("conv3_c")] = nn.Conv2d(512, 512, 3, 2, 1)
        #self.convs[("linear1_c")] = nn.Linear(3*512, 256)
        self.convs[("linear1_c")] = nn.Linear(2*512, 256)
        self.convs[("linear2_c")] = nn.Linear(256, 2*self.num_bins+4)
        #self.convs[("linear1_c_ada")] = nn.Linear(3*512, 256)
        #self.convs[("linear2_c_ada")] = nn.Linear(256, num_bins)

        self.convs[("relu1_c")] = nn.ReLU()
        self.convs[("relu2_c")] = nn.ReLU()
        self.convs[("relu3_c")] = nn.ReLU()
        self.convs[("relu4_c")] = nn.ReLU()
        self.convs[("relu5_c")] = nn.ReLU()
        self.convs[("sig_c")] = nn.Softmax(dim=1)#nn.Sigmoid()

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

        #self.zeros = nn.Parameter(torch.zeros(batch_size,1), requires_grad=False)

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

        xc = input_features[-1]
        batch_size, in_ch, _, _ = xc.size()
        x1 = self.convs[("relu1_c")](self.convs[("conv1_c")](xc))
        x2 = self.convs[("relu2_c")](self.convs[("conv2_c")](x1))
        x3 = self.convs[("relu3_c")](self.convs[("conv3_c")](x2))
        #print(x3.size())
        x4 = self.convs[("relu4_c")](self.convs[("linear1_c")](x3.reshape(batch_size, -1)))
        x5 = self.convs[("linear2_c")](x4)
        d_alpha = self.convs[("relu5_c")](x5[:,0:self.num_bins])
        d_d = self.convs[("relu5_c")](x5[:,self.num_bins:2*self.num_bins])
        #bin_width_sig = self.sigmoid(x5[:,self.num_bins:2*self.num_bins])
        bin_width = d_d/(torch.sum(d_d, dim=1, keepdim=True) + 1e-7)
        #d_camera_param = d_camera_param_x

        #bin_width = bin_width_sig/(torch.sum(bin_width_sig, dim=1, keepdim=True) + 1e-7)
        #bin_width = bin_width_sig
        camera_range = self.sigmoid(x5[:,2*self.num_bins:2*self.num_bins + 2])
        camera_offset = 2.0*(self.sigmoid(x5[:,2*self.num_bins + 2:2*self.num_bins + 4]) - 0.5)

        cam_lens_x = []
        for i in range(self.num_bins):
            lens_height = 0.0
            #for j in range(i, self.num_bins):
            for j in range(0, i+1):
                lens_height += d_alpha[:, j:j+1]
            cam_lens_x.append(lens_height)
        cam_lens_c = torch.cat(cam_lens_x, dim=1)
        camera_param = cam_lens_c*bin_width/(torch.sum(cam_lens_c*bin_width, dim=1, keepdim=True) + 1e-7)
        #camera_param = cam_lens_c*bin_width

        #camera_param = - cam_lens_c*bin_width*bin_width/(torch.sum(cam_lens_c*bin_width*bin_width, dim=1, keepdim=True) + 1e-7)
        #camera_param = cam_lens_c

        """
        bin_width_x = []
        for i in range(self.num_bins):
            lens_width = 0.0
            #for j in range(i, self.num_bins):
            for j in range(0, i+1):
                lens_width += d_bin_width[:, j:j+1]
            bin_width_x.append(lens_width)
        bin_width_c = torch.cat(bin_width_x, dim=1)
        #bin_width = bin_width_c/(torch.sum(bin_width_c, dim=1, keepdim=True) + 1e-7)
        bin_width = bin_width_c
        """
        #camera_param = cam_lens_c
        #return self.outputs, camera_param, bin_width, camera_range
        return self.outputs, camera_param, bin_width, camera_range, camera_offset
"""
class DepthDecoder_camera_ada3(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_bins=32, num_output_channels=1, use_skips=True):
        super(DepthDecoder_camera_ada3, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        self.num_bins = num_bins
        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.convs[("conv1_c")] = nn.Conv2d(512, 512, 3, 2, 1)
        self.convs[("conv2_c")] = nn.Conv2d(512, 512, 3, 2, 1)
        self.convs[("conv3_c")] = nn.Conv2d(512, 512, 3, 2, 1)
        self.convs[("linear1_c")] = nn.Linear(3*512, 256)
        self.convs[("linear2_c")] = nn.Linear(256, 2*self.num_bins+2)
        #self.convs[("linear1_c_ada")] = nn.Linear(3*512, 256)
        #self.convs[("linear2_c_ada")] = nn.Linear(256, num_bins)

        self.convs[("relu1_c")] = nn.ReLU()
        self.convs[("relu2_c")] = nn.ReLU()
        self.convs[("relu3_c")] = nn.ReLU()
        self.convs[("relu4_c")] = nn.ReLU()
        self.convs[("relu5_c")] = nn.ReLU()
        self.convs[("sig_c")] = nn.Softmax(dim=1)#nn.Sigmoid()

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

        #self.zeros = nn.Parameter(torch.zeros(batch_size,1), requires_grad=False)

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

        xc = input_features[-1]
        batch_size, in_ch, _, _ = xc.size()
        x1 = self.convs[("relu1_c")](self.convs[("conv1_c")](xc))
        x2 = self.convs[("relu2_c")](self.convs[("conv2_c")](x1))
        x3 = self.convs[("relu3_c")](self.convs[("conv3_c")](x2))
        #print(x3.size())
        x4 = self.convs[("relu4_c")](self.convs[("linear1_c")](x3.view(batch_size, -1)))
        x5 = self.convs[("linear2_c")](x4)
        d_camera_param = self.convs[("relu5_c")](x5[:,0:self.num_bins])
        bin_width_sig = self.sigmoid(x5[:,self.num_bins:2*self.num_bins])
        bin_width = bin_width_sig/(torch.sum(bin_width_sig, dim=1, keepdim=True) + 1e-7)
        #bin_width = bin_width_sig
        camera_range = self.sigmoid(x5[:,2*self.num_bins:2*self.num_bins + 2])

        cam_lens_x = []
        for i in range(self.num_bins):
            lens_height = 0.0
            for j in range(i, self.num_bins):
                lens_height += d_camera_param[:, j:j+1]
            cam_lens_x.append(lens_height)
        cam_lens_c = torch.cat(cam_lens_x, dim=1)
        camera_param = cam_lens_c/(torch.sum(cam_lens_c, dim=1, keepdim=True) + 1e-7)
        #camera_param = cam_lens_c
        return self.outputs, camera_param, bin_width, camera_range
"""
