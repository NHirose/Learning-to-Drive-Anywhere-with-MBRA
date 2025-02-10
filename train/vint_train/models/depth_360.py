import torch
import os
import vint_train.training.networks as networks
from vint_train.training.networks.layers import *

class Depth_est:
    def __init__(self, h_size, w_size, bin_size, batch_img, device, aug):
        self.h_size = h_size
        self.w_size = w_size
        self.device = device
        self.bin_size = bin_size
        self.batch_img = batch_img
        self.aug = aug
    
        self.enc_depth = networks.ResnetEncoder(18, True, num_input_images = 1)
        path = os.path.join("../deployment/model_weights/depthest_ploss/", "encoder.pth")
        model_dict = self.enc_depth.state_dict()
        pretrained_dict = torch.load(path, map_location=torch.device('cpu'))
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.enc_depth.load_state_dict(model_dict)
        self.enc_depth.eval().to(device)
    
        self.dec_depth = networks.DepthDecoder_camera_ada4(self.enc_depth.num_ch_enc, [0, 1, 2, 3], bin_size)
        path = os.path.join("../deployment/model_weights/depthest_ploss/", "depth.pth")
        model_dict = self.dec_depth.state_dict()
        pretrained_dict = torch.load(path, map_location=torch.device('cpu'))
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.dec_depth.load_state_dict(model_dict)
        self.dec_depth.eval().to(device)

        backproject_depth_fisheye = {}
        project_3d_fisheye = {}
        scale = 0
        h = h_size // (2 ** scale)
        w = w_size // (2 ** scale)
        self.backproject_depth_fisheye = BackprojectDepth_fisheye_inter_offset(batch_img, h, w, bin_size)
        self.backproject_depth_fisheye.to(device)

        self.lens_zero = torch.zeros((batch_img, 1)).to(self.device)
        self.binwidth_zero = torch.zeros((batch_img, 1)).to(self.device)        
                            
    def forward(self, image):
        image_batch = image
        
        with torch.no_grad():
            features = self.enc_depth(image_batch)
            outputs, camera_param_n, binwidth_n, camera_range_n, camera_offset_n = self.dec_depth(features)     
    
        if self.aug:
            camera_param = (camera_param_n[0:self.batch_img] + camera_param_n[self.batch_img:2*self.batch_img])/2.0
            binwidth = (binwidth_n[0:self.batch_img] + binwidth_n[self.batch_img:2*self.batch_img])/2.0
            camera_range = (camera_range_n[0:self.batch_img] + camera_range_n[self.batch_img:2*self.batch_img])/2.0
            camera_offset = (camera_offset_n[0:self.batch_img] + camera_offset_n[self.batch_img:2*self.batch_img])/2.0                                    
        else:
            camera_param = camera_param_n
            binwidth = binwidth_n
            camera_range = camera_range_n
            camera_offset = camera_offset_n
            
        cam_lens_x = []                    
        for i in range(self.bin_size):
            lens_height = torch.zeros(self.batch_img, 1, device=self.device)
            for j in range(0, i+1):
                lens_height += camera_param[0:self.batch_img, j:j+1]
            cam_lens_x.append(lens_height)
        cam_lens_c = torch.cat(cam_lens_x, dim=1)
        cam_lens = 1.0 - torch.cat([self.lens_zero, cam_lens_c], dim=1)

        lens_bincenter_x = []
        for i in range(self.bin_size):
            bin_center = torch.zeros(self.batch_img, 1, device=self.device)
            for j in range(0, i+1):
                bin_center += binwidth[0:self.batch_img, j:j+1]
            lens_bincenter_x.append(bin_center)
        lens_bincenter_c = torch.cat(lens_bincenter_x, dim=1)
        lens_bincenter = torch.cat([self.binwidth_zero, lens_bincenter_c], dim=1)

        if self.aug:
            dispf_x = outputs[("disp", 0)][0:self.batch_img]
            dispf_y = torch.flip(outputs[("disp", 0)][self.batch_img:2*self.batch_img], dims=[3])
            dispf = (dispf_x + dispf_y)/2.0 
        else:
            dispf = outputs[("disp", 0)][0:self.batch_img]

        pred_disp_f, depth_f = disp_to_depth(dispf, 0.1, 100.0)
        depth = depth_f

        eps = 1e-7
        lens_alpha = (cam_lens[:,1:self.bin_size+1] - cam_lens[:,0:self.bin_size])/(lens_bincenter[:,1:self.bin_size+1] - lens_bincenter[:,0:self.bin_size] + eps)
        lens_beta = (-cam_lens[:,1:self.bin_size+1]*lens_bincenter[:,0:self.bin_size] + cam_lens[:,0:self.bin_size]*lens_bincenter[:,1:self.bin_size+1] + eps)/(lens_bincenter[:,1:self.bin_size+1] - lens_bincenter[:,0:self.bin_size] + eps)
                 
        cam_points_f, h_back1, x_back1 = self.backproject_depth_fisheye(depth_f, lens_alpha[0:self.batch_img], lens_beta[0:self.batch_img], camera_range[0:self.batch_img], camera_offset[0:self.batch_img])

        return cam_points_f[:,0:3].reshape(self.batch_img,3,self.h_size,self.w_size), depth
