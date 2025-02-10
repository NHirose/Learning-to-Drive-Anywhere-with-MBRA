import sys
# appending a path
print(sys.path)
sys.path.append('/home/noriaki/Documents/data_auto_collection')
sys.path.insert(0, '/home/noriaki/Documents/data_auto_collection/yolov5')

import torch
import torchvision.transforms as T
import numpy as np

from utils_ds.parser import get_config
from utils_ds.draw import draw_boxes

from yolov5.utils.general import (
    check_img_size, non_max_suppression, scale_coords, xyxy2xywh)
from yolov5.utils.torch_utils import select_device, time_synchronized
from yolov5.utils.datasets import letterbox

from deep_sort import build_tracker
import matplotlib as mpl
import matplotlib.cm as cm

class Ped_est:
    def __init__(self, ):
        # ***************************** initialize DeepSORT **********************************
        cfg = get_config()
        cfg.merge_from_file("/home/noriaki/Documents/data_auto_collection/configs/deep_sort.yaml")
        cfg.DEEPSORT.REID_CKPT = "/home/noriaki/Documents/data_auto_collection/deep_sort/deep/checkpoint/ckpt.t7"
        device_tr = select_device("")
        use_cuda = device_tr.type != 'cpu' and torch.cuda.is_available()
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)

        # ***************************** initialize YOLO-V5 **********************************
        self.detector = torch.load('/home/noriaki/Documents/data_auto_collection/yolov5/weights/yolov5s.pt', map_location=device_tr)['model'].float()  # load to FP32
        self.detector.to(device_tr).eval()

        half = device_tr.type != 'cpu'
        if half:
            self.detector.half()  # to FP16

        names = self.detector.module.names if hasattr(self.detector, 'module') else self.detector.names
        self.transform_raw = T.Resize(size = (432,768))
        
    def forward(self, image_raw, image_depth, proj_3d):
        #TODO we may need to resize the double_x, double_y, double_z      
        #input:
        #image_raw: large image 3 x H x W (TODO we need to check BGR or RGB, and data type)
        
        proj_3d_resize = self.transform_raw(proj_3d)
        pc_x = proj_3d_resize[:,0:1]
        pc_y = proj_3d_resize[:,1:2]
        pc_z = proj_3d_resize[:,2:3]
        B, _, _, _ = image_raw.size()                
        #img0 = image_raw.cpu().numpy().astype(np.uint8).transpose(1, 2, 0)[:, :, ::-1]
        img0 = image_raw.cpu().numpy().astype(np.uint8).transpose(0, 2, 3, 1)[:, :, :, ::-1]
        img0_v = image_raw.cpu().numpy().astype(np.uint8).transpose(0, 2, 3, 1)
        #imgd = image_depth.cpu().numpy().astype(np.uint8).transpose(0, 2, 3, 1)[:, :, ::-1]
                
        img_torch_list = []
        
        for i in range(B):
            img = letterbox(img0[i], new_shape=1280)[0] #640
            #print("letterbox", img.shape)
        
            # Convert
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)

            # numpy to tensor
            img = torch.from_numpy(img).to("cuda")
            img = img.half()
            img /= 255.0  # 0 - 255 to 0.0 - 1.0

            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            s = '%gx%g ' % img.shape[2:]    # print string
            
            img_torch_list.append(img)
        img_torch = torch.cat(img_torch_list, axis=0)
        #print(img_torch.size())        
                
        with torch.no_grad():
            pred = self.detector(img_torch, augment=True)[0]  # list: bz * [ (#obj, 6)]

        # Apply NMS and filter object other than person (cls:0)
        pred = non_max_suppression(pred, 0.7, 0.5, classes=[0], agnostic=True)

        #print("pred", len(pred))
        image_list = []
        ped_list = []
        for i in range(B):
            # get all obj ************************************************************
            det = pred[i]  # for video, bz is 1
            if det is not None and len(det):  # det: (#obj, 6)  x1 y1 x2 y2 conf cls
                # Rescale boxes from img_size to original im0 size
                #print("??", img_torch.shape[2:], img0[i].shape)
                det[:, :4] = scale_coords(img_torch.shape[2:], det[:, :4], img0[i].shape).round()
        
                # Print results. statistics of number of each obj
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, self.detector.names[int(c)])  # add to string
        
                bbox_xywh = xyxy2xywh(det[:, :4]).cpu()
                confs = det[:, 4:5].cpu()
                # ****************************** deepsort ****************************
                #print("imgd", imgd[i].shape)
                outputs = self.deepsort.update(bbox_xywh, confs, img0[i])
                # (#ID, 5) x1,y1,x2,y2,track_ID
            else:
                outputs = torch.zeros((0, 5))
                    
            last_out = outputs

            ped_dict = {}
            if len(outputs) > 0:
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -1]
                img_d = draw_boxes(img0_v[i].copy(), bbox_xyxy, identities)  # BGR
                image_list.append(img_d)
                #depth_rgb = draw_boxes(depth_rgb.copy(), bbox_xyxy, identities)  # BGR
                    
                
                for ped in range(len(outputs)):
                    center_u = (bbox_xyxy[ped][1] + bbox_xyxy[ped][3])*0.5
                    center_v = (bbox_xyxy[ped][0] + bbox_xyxy[ped][2])*0.5
                    width_u = bbox_xyxy[ped][3]-bbox_xyxy[ped][1]
                    width_v = bbox_xyxy[ped][2]-bbox_xyxy[ped][0]
                    us = int(center_u - width_u*0.125)
                    ue = int(center_u + width_u*0.125)
                    vs = int(center_v - width_v*0.125)
                    ve = int(center_v + width_v*0.125)
                    ped_pixels_x = pc_x[i].cpu().numpy()[:, us:ue, vs:ve]
                    ped_pixels_y = pc_y[i].cpu().numpy()[:, us:ue, vs:ve]
                    ped_pixels_z = pc_z[i].cpu().numpy()[:, us:ue, vs:ve]                                                                                

                    #print(ped_pixels_x, ped_pixels_y, ped_pixels_z)
                    #print(pc_x[i].size(), pc_y[i].size(), pc_z[i].size())  
                    
                    if ped_pixels_x.size != 0 and ped_pixels_z.size != 0:             
                        xpos = np.median(ped_pixels_x)
                        zpos = np.median(ped_pixels_z) 
                        #xpos_min = np.min(ped_pixels_x)
                        #zpos_min = np.min(ped_pixels_z) 
                        x1, y1, x2, y2, idx = outputs[ped]
            
                        print("Ped. id", idx, "bbox", us, ue, vs, ve, "x, z position on cam coordinate", xpos, zpos)                       
                        ped_dict[idx] = [xpos.item(), zpos.item(), us, ue, vs, ve]
                    else:
                        print("zero array for ped estimation")
            else:
                image_list.append(img0_v[i]) 
            ped_list.append(ped_dict)                   
                                             
        return image_list, ped_list
