# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth

def transformation_from_parameters(axisangle, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M


def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T


def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)

        return cam_points

# For backprojection of fisheye (Noriaki Hirose)
class BackprojectDepth_fisheye(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(BackprojectDepth_fisheye, self).__init__()

        #for depth estimation from fisheye image view by Noriaki Hirose
        # fish_f (b,1,h,w): filter for fish eye image view. inside circle : 1, outside : 0
        # x_range (b,1,h,w): map of X for image coord. -1 ... 1
        # y_range (b,1,h,w): map of Y for image coord. -1 ... 1

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

        if self.height == self.width:
            hw = self.height
        else:
            print("This is not fisheye image!! fisheye image may be height == width")

        self.fisheye_f = np.zeros((hw,hw), dtype=np.float32)
        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        x_range = 2.0*np.array([range(0,hw)])/(hw - 1.0) - 1.0
        y_range = 1.0*np.transpose(2.0*np.array([range(0,hw)])/(hw - 1.0) - 1.0)
        x_rangex = x_range
        y_rangex = y_range
        #define matrix
        for i in range(hw-1):
            x_range = np.append(x_range, x_rangex, axis=0)
            y_range = np.append(y_range, y_rangex, axis=1)
        #fish eye filter
        for xi in range(hw):
            for yi in range(hw):
                if x_range[xi][yi]*x_range[xi][yi] + y_range[xi][yi]*y_range[xi][yi] <= 1.0:
                    self.fisheye_f[xi][yi] = 1.0

        self.fisheye_f = np.expand_dims(self.fisheye_f, axis=0)
        x_range = np.expand_dims(x_range, axis=0)
        y_range = np.expand_dims(y_range, axis=0)
        fisheye_bf = self.fisheye_f
        x_brange = x_range
        y_brange = y_range
        for i in range(batch_size-1):
            self.fisheye_f = np.append(self.fisheye_f, fisheye_bf, axis=0)
            x_range = np.append(x_range, x_brange, axis=0)
            y_range = np.append(y_range, y_brange, axis=0)

        self.fish_f = nn.Parameter(torch.from_numpy(np.expand_dims(self.fisheye_f, axis=1).astype(np.float32)).clone(), requires_grad=False)
        self.x_range = nn.Parameter(torch.from_numpy(np.expand_dims(x_range*self.fisheye_f, axis=1).astype(np.float32)).clone(), requires_grad=False)
        self.y_range = nn.Parameter(torch.from_numpy(np.expand_dims(y_range*self.fisheye_f, axis=1).astype(np.float32)).clone(), requires_grad=False)

    def forward(self, depth):
        cos_t = torch.sqrt(self.x_range**2 + self.y_range**2)
        XY_t = depth/torch.sqrt(1.0/cos_t**2 - 1.0 + self.eps)
        #XY_t = depth * cos_t

        cos_td = self.x_range/(cos_t + self.eps)
        sin_td = self.y_range/(cos_t + self.eps)    #0.000000000000001
        X_t = XY_t*cos_td
        Y_t = XY_t*sin_td
        Z_t = depth
        #Z_t = torch.sqrt(F.relu(depth*depth - X_t*X_t - Y_t*Y_t))

        cam_points = torch.cat([X_t, Y_t, Z_t], dim = 1).view(self.batch_size, 3, -1)
        cam_points = torch.cat([cam_points, self.ones], 1)

        return cam_points
# For backprojection of fisheye (Noriaki Hirose)

class BackprojectDepth_fisheye_inter(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width, bin_size, eps=1e-7):
        super(BackprojectDepth_fisheye_inter, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

        if self.height == self.width:
            hw = self.height
        else:
            print("This is not fisheye image!! fisheye image may be height == width")

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.ones_hw = nn.Parameter(torch.ones(self.batch_size), requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = nn.Parameter(self.pix_coords.repeat(batch_size, 1, 1).view(self.batch_size, 2, self.height, self.width), requires_grad=False)
        self.x_range_k = nn.Parameter(self.pix_coords[:,0:1,:,:], requires_grad=False)
        self.y_range_k = nn.Parameter(self.pix_coords[:,1:2,:,:], requires_grad=False)

        self.bin_size = bin_size
        self.sfmax = nn.Softmax(dim=2)

        self.cx = nn.Parameter(torch.ones((self.batch_size))*(self.width - 1.0)/2.0, requires_grad=False)
        self.cy = nn.Parameter(torch.ones((self.batch_size))*(self.height - 1.0)/2.0, requires_grad=False)

        self.relu = nn.ReLU()

    def forward(self, depth, alpha, beta, cam_range):

        index_x = cam_range[:,0:1] + self.eps
        index_y = cam_range[:,1:2] + self.eps

        #self.x_range = index_x.view(self.batch_size, 1, 1, 1)*(self.x_range_k - self.cx.view(self.batch_size, 1, 1, 1))/self.cx.view(self.batch_size, 1, 1, 1)
        #self.y_range = index_y.view(self.batch_size, 1, 1, 1)*(self.y_range_k - self.cy.view(self.batch_size, 1, 1, 1))/self.cy.view(self.batch_size, 1, 1, 1)
        self.x_range = (self.x_range_k - self.cx.view(self.batch_size, 1, 1, 1))/self.cx.view(self.batch_size, 1, 1, 1)
        self.y_range = (self.y_range_k - self.cy.view(self.batch_size, 1, 1, 1))/self.cy.view(self.batch_size, 1, 1, 1)

        xy_t = torch.sqrt(self.x_range**2 + self.y_range**2 + self.eps)
        bin_height_group = (alpha.unsqueeze(1).unsqueeze(1).unsqueeze(1))*(xy_t.unsqueeze(4)) + beta.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        bin_height_c, _ = torch.min(bin_height_group, dim=4)
        bin_height = torch.clamp(bin_height_c, min=0.0, max=1.0)

        XY_t = xy_t/(bin_height + self.eps)*depth

        cos_td = self.x_range/(xy_t + self.eps)
        sin_td = self.y_range/(xy_t + self.eps)

        X_t = XY_t*cos_td
        Y_t = XY_t*sin_td
        Z_t = depth

        cam_points = torch.cat([X_t, Y_t, Z_t], dim = 1).view(self.batch_size, 3, -1)
        cam_points = torch.cat([cam_points, self.ones], 1)

        return cam_points, bin_height, xy_t

class BackprojectDepth_fisheye_inter_offset(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width, bin_size, eps=1e-7):
        super(BackprojectDepth_fisheye_inter_offset, self).__init__()

        #for depth estimation from fisheye image view by Noriaki Hirose
        # fish_f (b,1,h,w): filter for fish eye image view. inside circle : 1, outside : 0
        # x_range (b,1,h,w): map of X for image coord. -1 ... 1
        # y_range (b,1,h,w): map of Y for image coord. -1 ... 1

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

        if self.height == self.width:
            hw = self.height
        else:
            print("This is not fisheye image!! fisheye image may be height == width")

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.ones_hw = nn.Parameter(torch.ones(self.batch_size), requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = nn.Parameter(self.pix_coords.repeat(batch_size, 1, 1).view(self.batch_size, 2, self.height, self.width), requires_grad=False)
        #self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
        #                               requires_grad=False)
        self.x_range_k = nn.Parameter(self.pix_coords[:,0:1,:,:], requires_grad=False)
        self.y_range_k = nn.Parameter(self.pix_coords[:,1:2,:,:], requires_grad=False)

        self.bin_size = bin_size
        self.sfmax = nn.Softmax(dim=2)

        self.cx = nn.Parameter(torch.ones((self.batch_size))*(self.width-1)/2.0, requires_grad=False)
        self.cy = nn.Parameter(torch.ones((self.batch_size))*(self.height-1)/2.0, requires_grad=False)

        self.relu = nn.ReLU()

    #def forward(self, depth, bins, binwidth, bin_X, alpha, beta, cam_range, cam_offset):
    def forward(self, depth, alpha, beta, cam_range, cam_offset):

        index_x = cam_range[:,0:1] + self.eps
        index_y = cam_range[:,1:2] + self.eps

        offset_x = cam_offset[:,0:1] + self.eps
        offset_y = cam_offset[:,1:2] + self.eps

        #print(self.batch_size, cam_range.size(), cam_offset.size())
        self.x_range = offset_x.view(self.batch_size, 1, 1, 1) + index_x.view(self.batch_size, 1, 1, 1)*(self.x_range_k - self.cx.view(self.batch_size, 1, 1, 1))/self.cx.view(self.batch_size, 1, 1, 1)
        self.y_range = offset_y.view(self.batch_size, 1, 1, 1) + index_y.view(self.batch_size, 1, 1, 1)*(self.y_range_k - self.cy.view(self.batch_size, 1, 1, 1))/self.cy.view(self.batch_size, 1, 1, 1)
        #self.x_range = (self.x_range_k - self.cx.view(self.batch_size, 1, 1, 1))/self.cx.view(self.batch_size, 1, 1, 1)
        #self.y_range = (self.y_range_k - self.cy.view(self.batch_size, 1, 1, 1))/self.cy.view(self.batch_size, 1, 1, 1)

        xy_t = torch.sqrt(self.x_range**2 + self.y_range**2 + self.eps)
        #print(xy_t.size())
        #print(beta.size())
        #print(alpha.size())
        bin_height_group = (alpha.unsqueeze(1).unsqueeze(1).unsqueeze(1))*(xy_t.unsqueeze(4)) + beta.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        bin_height_c, _ = torch.min(bin_height_group, dim=4)
        bin_height = torch.clamp(bin_height_c, min=-0.0, max=1.0)
        #bin_height = bin_height_c
        #bin_height = bin_height_group[:,:,:,:,0]

        #cos_t = torch.sqrt(self.x_range**2 + self.y_range**2 + self.eps)/torch.sqrt(self.x_range**2 + self.y_range**2 + bin_height**2 + self.eps)
        #XY_t = depth/torch.sqrt(1.0/(cos_t**2 + self.eps) - 1.0)
        XY_t = xy_t/(bin_height + self.eps)*depth
          
        cos_td = self.x_range/(xy_t + self.eps)
        sin_td = self.y_range/(xy_t + self.eps)

        X_t = XY_t*cos_td
        Y_t = XY_t*sin_td
        Z_t = depth

        cam_points = torch.cat([X_t, Y_t, Z_t], dim = 1).view(self.batch_size, 3, -1)
        cam_points = torch.cat([cam_points, self.ones], 1)

        return cam_points, bin_height, xy_t

class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        P = torch.matmul(K, T)[:, :3, :]

        cam_points = torch.matmul(P, points)

        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords

# For projection of fisheye (Noriaki Hirose)
class Project3D_fisheye(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D_fisheye, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, T):
        P = T[:, :3, :]

        cam_points = torch.matmul(P, points).view(self.batch_size, 3, self.height, self.width)
        X_s = cam_points[:,0,:,:]
        Y_s = cam_points[:,1,:,:]
        Z_s = cam_points[:,2,:,:]
        mask = (Z_s < 0).float()
        #print(mask.size(), Z_s.size())
        L_s = torch.sqrt(X_s**2 + Y_s**2 + Z_s**2)

        flow_x = X_s/(L_s + self.eps)
        flow_y = Y_s/(L_s + self.eps)
        pix_coords = torch.cat([flow_x.unsqueeze(3), flow_y.unsqueeze(3)], dim = 3)

        return pix_coords, mask
# For projection of fisheye (Noriaki Hirose)

class Project3D_fisheye_inter(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, batch_size, height, width, bin_size, eps=1e-7):
        super(Project3D_fisheye_inter, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps
        self.bin_size = bin_size
        self.sfmax = nn.Softmax(dim=3)

        self.ones_hw = nn.Parameter(torch.ones(self.batch_size), requires_grad=False)
        #self.relu = nn.ReLU()

    def forward(self, points, T, alpha, beta, cam_range):

        index_x = cam_range[:,0:1] + self.eps
        index_y = cam_range[:,1:2] + self.eps        
        
        P = T[:, :3, :]

        cam_points = torch.matmul(P, points).view(self.batch_size, 3, self.height, self.width)
        X_s = cam_points[:,0,:,:]
        Y_s = cam_points[:,1,:,:]
        Z_s = cam_points[:,2,:,:]
        mask = (Z_s < 0).float()
        ratio = Z_s/(torch.sqrt(X_s**2 + Y_s**2) + self.eps)

        xy_t_group = beta.unsqueeze(1).unsqueeze(1)/(ratio.unsqueeze(3) - alpha.unsqueeze(1).unsqueeze(1) + self.eps)
        height_group = ratio.unsqueeze(3)*beta.unsqueeze(1).unsqueeze(1)/(ratio.unsqueeze(3) -alpha.unsqueeze(1).unsqueeze(1) + self.eps)
        xy_t_c, _ = torch.min(xy_t_group, dim=3)
        height_t_c, _ = torch.min(height_group, dim=3)
        xy_t = torch.clamp(xy_t_c, min=0.0, max=1.0)
        height_t = torch.clamp(height_t_c, min=0.0, max=1.0)
       
        flow_x = (xy_t * (X_s/(torch.sqrt(X_s**2 + Y_s**2) + self.eps)))
        flow_y = (xy_t * (Y_s/(torch.sqrt(X_s**2 + Y_s**2) + self.eps)))

        pix_coords = torch.cat([flow_x.unsqueeze(3), flow_y.unsqueeze(3)], dim = 3)

        return pix_coords, mask, height_t, xy_t, Z_s, [X_s, Y_s, Z_s]

class Project3D_fisheye_inter_offset(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, batch_size, height, width, bin_size, eps=1e-7):
        super(Project3D_fisheye_inter_offset, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps
        self.bin_size = bin_size
        self.sfmax = nn.Softmax(dim=3)

        self.ones_hw = nn.Parameter(torch.ones(self.batch_size), requires_grad=False)
        #self.relu = nn.ReLU()

    #def forward(self, points, T, bins, bin_X, binwidth, alpha, beta, K):
    #def forward(self, points, T, bins, bin_X, binwidth, alpha, beta, cam_range, cam_offset):
    def forward(self, points, T, alpha, beta, cam_range, cam_offset):

        index_x = cam_range[:,0:1] + self.eps
        index_y = cam_range[:,1:2] + self.eps        

        offset_x = cam_offset[:,0:1] + self.eps
        offset_y = cam_offset[:,1:2] + self.eps  
        
        P = T[:, :3, :]

        cam_points = torch.matmul(P, points).view(self.batch_size, 3, self.height, self.width)
        X_s = cam_points[:,0,:,:] + self.eps 
        Y_s = cam_points[:,1,:,:] + self.eps 
        Z_s = cam_points[:,2,:,:] + self.eps 
        mask = (Z_s < 0).float()
        ratio = Z_s/(torch.sqrt(X_s**2 + Y_s**2) + self.eps)

        xy_t_group = beta.unsqueeze(1).unsqueeze(1)/(ratio.unsqueeze(3) - alpha.unsqueeze(1).unsqueeze(1) + self.eps)
        height_group = ratio.unsqueeze(3)*beta.unsqueeze(1).unsqueeze(1)/(ratio.unsqueeze(3) -alpha.unsqueeze(1).unsqueeze(1))
        #print(height_group)
        xy_t_c, _ = torch.min(xy_t_group, dim=3)
        xy_t = torch.clamp(xy_t_c, min=0.0, max=1.0)
        height_t, _ = torch.min(height_group, dim=3)
        #print(height_t)
        
        flow_x = ((xy_t * (X_s/(torch.sqrt(X_s**2 + Y_s**2) + self.eps))) - offset_x.view(self.batch_size, 1, 1))/index_x.view(self.batch_size, 1, 1)
        flow_y = ((xy_t * (Y_s/(torch.sqrt(X_s**2 + Y_s**2) + self.eps))) - offset_y.view(self.batch_size, 1, 1))/index_y.view(self.batch_size, 1, 1)

        pix_coords = torch.cat([flow_x.unsqueeze(3), flow_y.unsqueeze(3)], dim = 3)

        #return pix_coords, mask, height_t, xy_t
        return pix_coords, mask, height_t, xy_t, Z_s, [X_s, Y_s, Z_s]

class Project3D_fisheye_inter_offset_traj(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, batch_size, height, width, bin_size, eps=1e-7):
        super(Project3D_fisheye_inter_offset_traj, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps
        self.bin_size = bin_size
        self.sfmax = nn.Softmax(dim=3)

        self.ones_hw = nn.Parameter(torch.ones(self.batch_size), requires_grad=False)
        #self.relu = nn.ReLU()

    #def forward(self, points, T, bins, bin_X, binwidth, alpha, beta, K):
    #def forward(self, points, T, bins, bin_X, binwidth, alpha, beta, cam_range, cam_offset):
    def forward(self, points, alpha, beta, cam_range, cam_offset):

        index_x = cam_range[:,0:1] + self.eps
        index_y = cam_range[:,1:2] + self.eps        

        offset_x = cam_offset[:,0:1] + self.eps
        offset_y = cam_offset[:,1:2] + self.eps  

        X_s = points[:,0,:] + self.eps 
        Y_s = points[:,1,:] + self.eps 
        Z_s = points[:,2,:] + self.eps 
        mask = (Z_s < 0).float()
        ratio = Z_s/(torch.sqrt(X_s**2 + Y_s**2) + self.eps)

        print(beta.size(), ratio.size(), alpha.size())
        
        xy_t_group = beta.unsqueeze(1)/(ratio.unsqueeze(2) - alpha.unsqueeze(1) + self.eps)
        #xy_t_group = beta.unsqueeze(1).unsqueeze(1)/(ratio.unsqueeze(3) - alpha.unsqueeze(1).unsqueeze(1) + self.eps)        
        #height_group = ratio.unsqueeze(3)*beta.unsqueeze(1).unsqueeze(1)/(ratio.unsqueeze(3) -alpha.unsqueeze(1).unsqueeze(1))
        #print(height_group)
        xy_t_c, _ = torch.min(xy_t_group, dim=2)
        xy_t = torch.clamp(xy_t_c, min=0.0, max=1.0)
        #height_t, _ = torch.min(height_group, dim=3)
        #print(height_t)
        
        flow_x = ((xy_t * (X_s/(torch.sqrt(X_s**2 + Y_s**2) + self.eps))) - offset_x.view(self.batch_size, 1, 1))/index_x.view(self.batch_size, 1, 1)
        flow_y = ((xy_t * (Y_s/(torch.sqrt(X_s**2 + Y_s**2) + self.eps))) - offset_y.view(self.batch_size, 1, 1))/index_y.view(self.batch_size, 1, 1)
        #print("before", flow_x.size(), flow_y.size())
        
        #flow_x = (xy_t * (X_s/(torch.sqrt(X_s**2 + Y_s**2) + self.eps)))
        #flow_y = (xy_t * (Y_s/(torch.sqrt(X_s**2 + Y_s**2) + self.eps)))
        #print("after", flow_x.size(), flow_y.size())
                
        #pix_coords = torch.cat([flow_x.unsqueeze(3), flow_y.unsqueeze(3)], dim = 3)
        #return pix_coords, mask, height_t, xy_t
        return flow_x, flow_y, X_s, Y_s, Z_s


class Project3D_fisheye_inter_footprint(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, batch_size, height, width, bin_size, eps=1e-7):
        super(Project3D_fisheye_inter_footprint, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps
        self.bin_size = bin_size
        self.sfmax = nn.Softmax(dim=3)

        self.ones_hw = nn.Parameter(torch.ones(self.batch_size), requires_grad=False)
        #self.relu = nn.ReLU()

    def forward(self, points, alpha, beta, cam_range):

        index_x = cam_range[:,0:1] + self.eps
        index_y = cam_range[:,1:2] + self.eps        
        
        X_s = points[:, :, 0]
        Y_s = points[:, :, 1]
        Z_s = points[:, :, 2]
        mask = (Z_s < 0).float()
        ratio = Z_s/(torch.sqrt(X_s**2 + Y_s**2) + self.eps)
        #print("ratio", ratio.size())
        #print("alpha", alpha.size())
        #print("beta", beta.size())

        xy_t_group = beta.unsqueeze(1)/(ratio.unsqueeze(2) - alpha.unsqueeze(1) + self.eps)
        #height_group = ratio.unsqueeze(3)*beta.unsqueeze(1).unsqueeze(1)/(ratio.unsqueeze(3) -alpha.unsqueeze(1).unsqueeze(1) + self.eps)
        xy_t_c, _ = torch.min(xy_t_group, dim=2)
        #height_t_c, _ = torch.min(height_group, dim=3)
        xy_t = torch.clamp(xy_t_c, min=0.0, max=1.0)
        #height_t = torch.clamp(height_t_c, min=0.0, max=1.0)
       
        flow_x = (xy_t * (X_s/(torch.sqrt(X_s**2 + Y_s**2) + self.eps)))
        flow_y = (xy_t * (Y_s/(torch.sqrt(X_s**2 + Y_s**2) + self.eps)))

        return flow_x, flow_y

def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")


def get_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()

def get_smooth_loss_mask(disp, img, mask):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    mask_x = mask[:, :, :, :-1] * mask[:, :, :, 1:]
    mask_y = mask[:, :, :-1, :] * mask[:, :, 1:, :]

    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return (mask_x*grad_disp_x).mean() + (mask_y*grad_disp_y).mean()

class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)

def compute_depth_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    sq_rel = torch.mean((gt - pred) ** 2 / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3
