#!/usr/bin/env python
# -*- coding: utf-8 -*-
# From Seigo Ito topic no doki

DIR_loc = "/mnt/sdb/earth-rovers-sdk"
#DIR_loc = "/home/vizbot/earth-rovers-sdk"

import sys

#index = sys.path.index('/home/vizbot/visualnav-transformer/train')
#print(index)
sys.path.remove('/home/vizbot/visualnav-transformer/train')
#print(sys.path)

sys.path.insert(0, DIR_loc + '/train')
#sys.path.insert(0, '/home/vizbot/visualnav-transformer/train')
sys.path.insert(0, '/home/vizbot/DeepSORT_YOLOv5_Pytorch')
sys.path.insert(0, '/usr/lib/python3.8/site-packages/torch2trt-0.4.0-py3.8.egg')
#sys.path.insert(0, '/home/vizbot/.local/lib/python3.8/site-packages/torch-1.13.0a0+340c4120.nv22.06.dist-info')

"""
sys.path.insert(0, '/media/vizbot/logging_usb/pedest')
sys.path.insert(0, '/usr/local/lib/python3.6/dist-packages/torchvision-0.9.0a0+01dfa8e-py3.6-linux-aarch64.egg')
"""
from yolov5.utils.general import (
    check_img_size, non_max_suppression, scale_coords, xyxy2xywh)
from yolov5.utils.torch_utils import select_device, time_synchronized
from yolov5.utils.datasets import letterbox

from utils_ds.parser import get_config
from utils_ds.draw import draw_boxes
from deep_sort import build_tracker
import time

from scipy.spatial.transform import Rotation as R
import os

#ROS
import rospy
import message_filters
from geometry_msgs.msg import Twist, Pose
from sensor_msgs.msg import Image, CompressedImage, JointState, LaserScan, CompressedImage, Imu
from std_msgs.msg import Float32
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from nav_msgs.msg import Odometry
#from ar_track_alvar_msgs.msg import AlvarMarkers
from std_msgs.msg import Int32
from nav_msgs.msg import Odometry

from std_msgs.msg import Bool
from create_msgs.msg import Bumper
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import rosparam
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

#PIL
#from utils.PIL2CV import pil2cv, cv2pil
#import ImageDraw
from PIL import ImageDraw
#import Image as PILImage
from PIL import Image as PILImage

#NN model
#from vunet import VUNet
#from polinet import PoliNet, PoliNet_geo2_trav_vis, ImageLoss, InputRegularizationLoss, PoliNet_sn_RL_feature
#from polinet import Polinet_feature, Polinet_critic, Polinet_actor2
#from pednet_x import PedNet_delta_small

#torch
import torch
import torch.nn.functional as F
import yaml

import cv2
import rosbag

#for chatting with Facebook
import fbchat
from fbchat.models import *
import re

#others
from cv_bridge import CvBridge, CvBridgeError
import sys
import cv2
import time
import numpy as np
import math
import numpy
from datetime import date, datetime

import json

import tensor_dict_convert
import tensor_dict_msgs.msg as tm
from transforms3d.quaternions import mat2quat
from transforms3d.euler import euler2mat, mat2euler
from transforms3d import quaternions

#import networks
from layers import *
import torchvision.transforms as T
#from torch2trt import torch2trt

import pickle 
import random

import matplotlib.pyplot as plt
from transforms3d.euler import euler2mat, mat2euler
import transforms3d

#from NoMad
from utils_policy import msg_to_pil, to_numpy, transform_images, transform_images_exaug, load_model, transform_images_lelan, transform_images_noriaki3

import clip
#from PIL import Image

currentUrl = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(currentUrl, 'yolov5')))

ini = 0
count_safe = 0
count = 0
Lpast = 1.0
vel_res = 0.0
wvel_res = 0.0
vel_ref = 0.0
wvel_ref = 0.0
Lth = 0.18

swopt = 1

image360 = True
fisheye = False
rsense = False

store_hist = 0
init_hist = 0
image_hist = []

now = datetime.now()
d4 = now.strftime("%m-%d-%Y_%H:%M:%S")
newpath = "/mnt/sdb/eval_RL/" + d4
if not os.path.exists(newpath):
    os.makedirs(newpath)

# load model weights
model_config_path = "/mnt/sdb/earth-rovers-sdk/config/frodbot_dist_IL2_gps.yaml"

ckpth_path = "/mnt/sdb/earth-rovers-sdk/policy/MBL_based_gps/latest_b.pth"
#ckpth_path = "/mnt/sdb/earth-rovers-sdk/policy/IL2_gps/latest.pth"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

with open(model_config_path, "r") as f:
    model_params = yaml.safe_load(f)

if os.path.exists(ckpth_path):
    print(f"Loading model from {ckpth_path}")
else:
    raise FileNotFoundError(f"Model weights not found at {ckpth_path}")
    
model = load_model(
    ckpth_path,
    model_params,
    device,
)
model = model.to(device)
model.eval()  

pytorch_total_params = sum(p.numel() for p in model.parameters())
print("Num of params:", pytorch_total_params)

flag_once = 0

nz = 100
ratio = 0.1

#center of picture
#yoko
xc = 310
#tate
yc = 321

yoffset = 310 
xoffset = 310
xyoffset = 280
xplus = 661
XYf = [(xc-xyoffset, yc-xyoffset), (xc+xyoffset, yc+xyoffset)]
XYb = [(xc+xplus-xyoffset, yc-xyoffset), (xc+xplus+xyoffset, yc+xyoffset)]

# resize parameters
rsizex = 128
rsizey = 128

rsizex_lnp = 96
rsizey_lnp = 96

#mask for 360 degree image
mask_br360 = np.loadtxt(open(os.path.join(os.path.dirname(__file__), "utils/mask_360view.csv"), "rb"), delimiter=",", skiprows=0)
#print mask_br.shape
mask_brr = mask_br360.reshape((1,1,128,256)).astype(np.float32)
mask_brr1 = mask_br360.reshape((1,128,256)).astype(np.float32)
mask_brrc = np.concatenate((mask_brr1, mask_brr1, mask_brr1), axis=0)

#mask for fisheye degree image
mask_recon = np.zeros((1, 128, 416), dtype = 'float32')
center_h = 0.5*128 - 0.5
center_w = 0.5*416 - 0.5
for i in range(416):
    for j in range(128):
        if ((i - center_w)**2)/(0.5*416*0.95)**2 + ((j - center_h)**2)/(0.5*128*1.5)**2 <= 1:
            mask_recon[0,j,i] = 1.0
                
mask_recon_batch = torch.from_numpy(mask_recon).float().clone().to("cuda").unsqueeze(0).repeat(1, 3, 1, 1)
mask_recon_polinet = {}
for scale in [0]:
    hsc = 128 // (2 ** scale)
    mask_recon_polinet[scale] = F.interpolate(mask_recon_batch, (hsc, hsc), mode='bilinear', align_corners=False)

mask = torch.zeros([1, 128,416])
for i in range(416):
    for j in range(128):
        if ((i - 208)**2)/208**2 + ((j - 64)**2)/64**2 < 1.0:
            mask[:, j, i] = 1.0
"""
mask_cpu = torch.zeros([2*xyoffset,2*xyoffset,1])            
for i in range(2*xyoffset):
    for j in range(2*xyoffset):
        if ((i - xyoffset)**2)/xyoffset**2 + ((j - xyoffset)**2)/xyoffset**2 < 1.0:
            mask_cpu[j, i, :] = 1.0

mask_cpu = mask_cpu.repeat(1,1,3)
"""
mask_gpu = mask.repeat(3,1,1).unsqueeze(dim=0).cuda()

Lmin = 100
i = 0
j = 0

#prev /cmd_vel
prev_cmd_vel = Twist()
prev_cmd_vel.linear.x = 0.0
prev_cmd_vel.linear.y = 0.0
prev_cmd_vel.linear.z = 0.0
prev_cmd_vel.angular.x = 0.0
prev_cmd_vel.angular.y = 0.0
prev_cmd_vel.angular.z = 0.0

# Function to compute the inverse of a quaternion
def quat_inverse(q):
    # Inverse of quaternion q = [x, y, z, w] is [-x, -y, -z, w]
    x, y, z, w = q
    return [-x, -y, -z, w]

# Function to compute quaternion multiplication
def quat_multiply(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return [
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    ]

# Function to calculate yaw from a quaternion
def quat_to_yaw(q):
    x, y, z, w = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y**2 + z**2))
    
def calc_relative_pose(pose_a, pose_b):
    """
    Compute the relative pose of pose_b with respect to pose_a.
    
    Parameters:
    pose_a: (x, y, theta) - Reference pose
    pose_b: (x, y, theta) - Target pose
    
    Returns:
    (dx, dy, dtheta) - The relative pose of pose_b in pose_a's frame
    """
    x_a, y_a, theta_a = pose_a
    x_b, y_b, theta_b = pose_b

    # Compute the relative translation
    dx = x_b - x_a
    dy = y_b - y_a

    # Rotate the translation into the local frame of pose_a
    dx_rel = np.cos(-theta_a) * dx - np.sin(-theta_a) * dy
    dy_rel = np.sin(-theta_a) * dx + np.cos(-theta_a) * dy

    # Compute the relative rotation
    dtheta = theta_b - theta_a
    dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi  # Normalize to [-pi, pi]

    return [dx_rel, dy_rel, dtheta]

def sinc_apx(angle):
    return torch.sin(3.141592*angle + 0.000000001)/(3.141592*angle + 0.000000001)

def twist_to_pose_diff(v, w, dt):
    """integrate 2D twist to get pose difference.

    Assuming constant velocity during time period `dt`.

    Args:
        v (float): velocity
        w (float): angular velocity
        dt (float): time delta

    """

    theta = -w  * dt
    z = v * dt * np.sinc(-theta / np.pi)
    x = -v * dt * np.sinc(-theta / (2 * np.pi)) * np.sin(-theta / 2)
    return x, z, theta

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

def twists_to_poses_mat(twists):
    px = []
    pz = []
    p = []
    Tacc = np.eye(4)
    for i in range(8):
        x, z, yaw = twist_to_pose_diff(twists[2*i], twists[2*i+1], 0.333)
        #Todom = Tr.of_pos_euler((x, y, 0.0), (0.0, 0.0, yaw))
        Todom = np.zeros((4, 4))
        Todom[0, 0] = np.cos(yaw)
        Todom[0, 2] = np.sin(yaw)
        Todom[1, 1] = 1.0
        Todom[2, 0] = -np.sin(yaw)
        Todom[2, 2] = np.cos(yaw)
        Todom[0, 3] = x #weighting for position
        Todom[2, 3] = z #weighting for position
        Todom[3, 3] = 1.0        
        
        Tacc = Tacc @ Todom
        p.append(Tacc)        
        px.append(Tacc[0, 3])
        pz.append(Tacc[2, 3])        
    return p, px, pz

def twists_to_poses_mat_torch(twists):
    px = []
    pz = []
    p = []
    bsl, _ = twists.size()
    Tacc = torch.eye(4, 4).unsqueeze(0).repeat(bsl,1,1)
    #print("init", Tacc.size())
    for i in range(8):
        x, z, yaw = twist_to_pose_diff_torch(twists[:, 2*i], twists[:, 2*i+1], 0.333)
        #Todom = Tr.of_pos_euler((x, y, 0.0), (0.0, 0.0, yaw))
        Todom = torch.zeros((bsl, 4, 4))
        Todom[:, 0, 0] = torch.cos(yaw)
        Todom[:, 0, 2] = torch.sin(yaw)
        Todom[:, 1, 1] = 1.0
        Todom[:, 2, 0] = -torch.sin(yaw)
        Todom[:, 2, 2] = torch.cos(yaw)
        Todom[:, 0, 3] = x #weighting for position
        Todom[:, 2, 3] = z #weighting for position
        Todom[:, 3, 3] = 1.0        
        
        #Tacc = Tacc @ Todom
        Tacc = torch.matmul(Tacc, Todom)
        
        #print(Tacc.size())
        Taccd = Tacc.clone()
        Taccd[:, 0, 3] = 2.0*Tacc.clone()[:, 0, 3]
        Taccd[:, 2, 3] = 2.0*Tacc.clone()[:, 2, 3]        
        p.append(Taccd.unsqueeze(1))        
        px.append(Tacc[:, 0, 3].unsqueeze(1))
        pz.append(Tacc[:, 2, 3].unsqueeze(1))        
    return p, px, pz

batch_data = {}
#lattice_mse = torch.nn.MSELoss()
lattice_mse = torch.nn.MSELoss(size_average=False)

#motion premitive
vel_00_00 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
vel_05_00 = [0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0]
vel_05_p03 = [0.5, 0.3, 0.5, 0.3, 0.5, 0.3, 0.5, 0.3, 0.5, 0.3, 0.5, 0.3, 0.5, 0.3, 0.5, 0.3]
vel_05_p06 = [0.5, 0.6, 0.5, 0.6, 0.5, 0.6, 0.5, 0.6, 0.5, 0.6, 0.5, 0.6, 0.5, 0.6, 0.5, 0.6]
vel_05_p09 = [0.5, 0.9, 0.5, 0.9, 0.5, 0.9, 0.5, 0.9, 0.5, 0.9, 0.5, 0.9, 0.5, 0.9, 0.5, 0.9]
vel_05_m03 = [0.5, -0.3, 0.5, -0.3, 0.5, -0.3, 0.5, -0.3, 0.5, -0.3, 0.5, -0.3, 0.5, -0.3, 0.5, -0.3]
vel_05_m06 = [0.5, -0.6, 0.5, -0.6, 0.5, -0.6, 0.5, -0.6, 0.5, -0.6, 0.5, -0.6, 0.5, -0.6, 0.5, -0.6]
vel_05_m09 = [0.5, -0.9, 0.5, -0.9, 0.5, -0.9, 0.5, -0.9, 0.5, -0.9, 0.5, -0.9, 0.5, -0.9, 0.5, -0.9]
vel_02_00 = [0.2, 0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 0.0]
vel_02_p03 = [0.2, 0.3, 0.2, 0.3, 0.2, 0.3, 0.2, 0.3, 0.2, 0.3, 0.2, 0.3, 0.2, 0.3, 0.2, 0.3]
vel_02_p06 = [0.2, 0.6, 0.2, 0.6, 0.2, 0.6, 0.2, 0.6, 0.2, 0.6, 0.2, 0.6, 0.2, 0.6, 0.2, 0.6]
vel_02_p09 = [0.2, 0.9, 0.2, 0.9, 0.2, 0.9, 0.2, 0.9, 0.2, 0.9, 0.2, 0.9, 0.2, 0.9, 0.2, 0.9]
vel_02_m03 = [0.2, -0.3, 0.2, -0.3, 0.2, -0.3, 0.2, -0.3, 0.2, -0.3, 0.2, -0.3, 0.2, -0.3, 0.2, -0.3]
vel_02_m06 = [0.2, -0.6, 0.2, -0.6, 0.2, -0.6, 0.2, -0.6, 0.2, -0.6, 0.2, -0.6, 0.2, -0.6, 0.2, -0.6]
vel_02_m09 = [0.2, -0.9, 0.2, -0.9, 0.2, -0.9, 0.2, -0.9, 0.2, -0.9, 0.2, -0.9, 0.2, -0.9, 0.2, -0.9]
#vel_02_m20 = [0.2, -2.0, 0.2, -2.0, 0.2, -2.0, 0.2, -2.0, 0.2, -2.0, 0.2, -2.0, 0.2, -2.0, 0.2, -2.0]

vel_prem = torch.Tensor([vel_00_00, vel_05_00, vel_05_p03, vel_05_p06, vel_05_p09, vel_05_m03, vel_05_m06, vel_05_m09, vel_02_00, vel_02_p03, vel_02_p06, vel_02_p09, vel_02_m03, vel_02_m06, vel_02_m09]).float().to("cuda")
print("vel_prem", vel_prem.size())
bvp, _ = vel_prem.size()
"""
bvp, _ = vel_prem.size()
ry = torch.zeros((bvp)).float().to("cuda")
pz = torch.zeros((bvp)).float().to("cuda")
px = torch.zeros((bvp)).float().to("cuda")        
        
ry_list = []
pz_list = []
px_list = []
        
for j in range(8):
    ry = ry - vel_prem[:, 2*j+1] * 0.33
    pz = pz + vel_prem[:, 2*j] * 0.33 * sinc_apx(-ry/(1.0*3.1415))
    px = px - vel_prem[:, 2*j] * 0.33 * sinc_apx(-ry/(2*3.1415)) * torch.sin(-ry / 2.0)
    ry_list.append(ry.unsqueeze(1))                        
    pz_list.append(pz.unsqueeze(1))  
    px_list.append(px.unsqueeze(1))                                  

ry_list_cat = torch.cat(ry_list, axis=1)
pz_list_cat = torch.cat(pz_list, axis=1)
px_list_cat = torch.cat(px_list, axis=1)


Tprem = torch.zeros((bvp, 4, 4))

print("ry", ry)        
Tprem[:, 0, 0] = torch.cos(ry)
Tprem[:, 0, 2] = torch.sin(ry)
Tprem[:, 1, 1] = 1.0
Tprem[:, 2, 0] = -torch.sin(ry)
Tprem[:, 2, 2] = torch.cos(ry)
Tprem[:, 0, 3] = 2.0*px #weighting for position
Tprem[:, 2, 3] = 2.0*pz #weighting for position
Tprem[:, 3, 3] = 1.0
            
p, px_np, pz_np = twists_to_poses_mat(vel_05_p10)            
"""

p_torch, px_torch, pz_torch = twists_to_poses_mat_torch(vel_prem)  
Tprem = torch.cat(p_torch, axis=1)
print(p_torch[0].size(), Tprem.size())
pz_torch_cat = torch.cat(pz_torch, axis=1).clone()
px_torch_cat = torch.cat(px_torch, axis=1).clone()
            
#for i in range(bvp):
#    #print(px_list_cat[i].cpu().numpy(), pz_list_cat[i].cpu().numpy())
#    #plt.plot(px_list_cat[i].cpu().numpy(), pz_list_cat[i].cpu().numpy(), marker = "o")
#    plt.plot(px_torch_cat[i].cpu().numpy(), pz_torch_cat[i].cpu().numpy(), marker = "x")
    
#plt.plot(px_np, pz_np, marker = "x")
    
#plt.show()    
    
"""
#for chatting with Facebook
#username = "vizbot.berkeley@gmail.com"
#password = "rail-4321" #"rail-1234"
#username = "kosuke11282016@gmail.com"
#password = "railrail1234"
username = "noriaki.hirose@gmail.com"
password = "0326nori"

client = fbchat.Client(username, password)
#cookies = client.getSession()
#with open("session.json", "w") as f:
#    json.dump(cookies, f)

no_of_friends = 1
users = client.fetchThreadList()
thread_type = ThreadType.USER
path_image = r'/home/vizbot/Pictures/test.png'

name = u'Noriaki  Hirose'
friends = client.searchForUsers(name)  # return a list of names
friend = friends[0]
msg_user = "Vizbot was stucked at this point. Please help me!"
msg_user2 = "Vizbot was out of map area. Please help me!"
"""
#for initial velocity for teleop helping
vjoy = 0.0
wjoy = 0.0

#data collection
today = date.today()
d4 = today.strftime("%b-%d-%Y")
#newpath = "/media/vizbot/logging_usb/dataset_auto_collection" + "/" + d4
#newpath = "/home/vizbot/dataset_auto_collection" + "/" + d4
#newpath = "/mnt/sdb/dataset_auto_collection" + "/" + d4

if not os.path.exists(newpath):
    os.makedirs(newpath)
    os.makedirs(newpath + '/tmp')
    #os.makedirs(newpath + "/360img")
    #os.makedirs(newpath + "/fisheye")
    #os.makedirs(newpath + "/odom")
#ofs1=open(newpath + "/360img" + ".txt","w")
#ofs2=open(newpath + "/fisheye" + ".txt","w")
#ofs3=open(newpath + "/odom" + ".txt","w")

fcount = 0
# Iterate directory
for path in os.listdir(newpath):
    # check if current path is a file
    if os.path.isfile(os.path.join(newpath, path)):
        fcount += 1

icount = 0
#topic_fisheye = Image()
#topic_spherical = Image()
topic_fisheye = CompressedImage()
topic_spherical = CompressedImage()
goal_img_topic = Image()
topic_odom = Odometry()
topic_laserscan = LaserScan()
#topic_armarker = AlvarMarkers()
topic_bumper = Bumper()
topic_velcmd = Twist()
topic_goalID = Int32()
markerArray = MarkerArray()
topic_t265 = Odometry()
topic_odomt = Odometry()
topic_trans_mat = Pose()
topic_gpose_mat = Pose()

bag_file = rosbag.Bag(newpath + '/' + str(fcount).zfill(8) + '.bag', 'w')

def unddp_state_dict(state_dict):
    """convert DDP-trained module's state dict to raw module's.

    Nothing is done if state_dict isn't DDP-trained.

    """

    if not all([s.startswith("module.") for s in state_dict.keys()]):
        return state_dict

    return OrderedDict((k[7:], v) for k, v in state_dict.items())

class Pano_image:
    def __init__(self, height, width, batch_size):
        self.height = height
        self.width = width
        pix_coords = np.zeros((self.height, self.width, 2), dtype=np.float32)
        for h in range(self.height):
            for w in range(self.width):
                # panorama back projection
                theta = math.pi * ((w + 0.5) / (self.width / 2) - 0.5)
                phi = math.pi * ((h + 0.5) / self.height - 0.5)
                
                x = math.sin(theta) * math.cos(phi)
                y = math.sin(phi)
                z = math.cos(theta) * math.cos(phi)

                # projection on fisheye
                if w < self.width / 2:
                    r = 2 * math.acos(z) / math.pi
                    fish = math.atan2(y,x)
                    pix_coords[h, w, 0] = 0.25 * r * math.cos(fish) + 0.25 - 0.5 / self.width
                    pix_coords[h, w, 1] = 0.5 * r * math.sin(fish) + 0.5 - 0.5 / self.height
                else:
                    r = 2 * math.acos(-z) / math.pi
                    fish = math.atan2(-y, -x)
                    pix_coords[h, w, 0] = 0.25 * r * math.cos(fish) + 0.75 - 0.5 / self.width
                    pix_coords[h, w, 1] = -0.5 * r * math.sin(fish) + 0.5 - 0.5 / self.height
                    
        pix_coords = (pix_coords - 0.5) * 2
        pix_coords = torch.from_numpy(pix_coords)
        pix_coords = pix_coords.repeat(batch_size, 1, 1, 1)
        
        self.pix_coords = nn.Parameter(pix_coords, requires_grad=False).cuda()

    def forward(self, image):
        panorama = F.grid_sample(image, self.pix_coords, padding_mode="border")
        return panorama

class TrainingRosInterface:
    def __init__(self, ):
        #rospy.init_node("training_node")

        #self.rb_queue = rb_queue
        #self.param_queue = param_queue

        self.rotmat_tf = np.array([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)

        """
        self.param_publisher = rospy.Publisher(
            rospy.get_param("~param_topic", "/actor_params"),
            TensorDict,
            queue_size=1,
            latch=True,
        )
        """
        self.rb_subscribe = rospy.Subscriber(
            rospy.get_param("~rb_topic", "/replay_buffer_data"),
            tm.TensorDict,
            self.rb_callback,
        )

        #self.param_publish_callback = rospy.Timer(rospy.Duration(2), self.param_pub_callback)
        self.buffer_loc = []
        self.prev_mat = np.eye(4)
        self.prev_t265 = np.eye(4)

        self.counter = 0
        self.flag_jump = 0
        self.flag_init = 0
        
        prev_trans = np.eye(4, dtype=np.float32)

    def sinc_apx_np(self, angle):
        return np.sin(3.141592*angle + 0.000000001)/(3.141592*angle + 0.000000001)

    """
    def func_pose_reward(self, pred_poses, Tref_local):
        ry = 0.0
        pz = 0.0
        px = 0.0

        Todom_list = []
        weight_position = 2.0
        step_length = 8                
        for j in range(step_length):
            if j < step_length:
                ry = ry - pred_poses[0, 2*j+1, 0, 0] * 0.33
                pz = pz + pred_poses[0, 2*j, 0, 0] * 0.33 * self.sinc_apx_np(-ry/3.1415)
                px = px - pred_poses[0, 2*j, 0, 0] * 0.33 * self.sinc_apx_np(-ry/(2*3.1415)) * np.sin(-ry / 2.0)
                            
                TodomM = np.zeros((1, 4, 4))                                          

                TodomM[0, 0, 0] = np.cos(ry)
                TodomM[0, 0, 2] = np.sin(ry)
                TodomM[0, 1, 1] = 1.0
                TodomM[0, 2, 0] = -np.sin(ry)
                TodomM[0, 2, 2] = np.cos(ry)
                TodomM[0, 0, 3] = weight_position*px #weighting for position
                TodomM[0, 2, 3] = weight_position*pz #weighting for positionexport ROS_IP=128.32.175.73
                TodomM[0, 3, 3] = 1.0
                    
                Todom_list.append(TodomM)
                    
        Todom_listd = np.concatenate(Todom_list, axis=0)            
        TodomN = np.zeros((4, 4))
        
        TodomN[0, 0] = np.cos(ry)
        TodomN[0, 2] = np.sin(ry)
        TodomN[1, 1] = 1.0
        TodomN[2, 0] = -np.sin(ry)
        TodomN[2, 2] = np.cos(ry)
        TodomN[0, 3] = weight_position*px #weighting for position
        TodomN[2, 3] = weight_position*pz #weighting for position
        TodomN[3, 3] = 1.0

        px2 = Tref_local[0, 3]/weight_position
        pz2 = Tref_local[2, 3]/weight_position     
        cos2 = Tref_local[0, 0]
        sin2 = Tref_local[0, 2]

        #virtual 8th pose        
        #px1 = px
        #pz1 = pz
        #cos1 = np.cos(ry)
        #sin1 = np.sin(ry)

        #xcom = (weight_position**2)*(2.0*px1 - 2.0*px2) * sin1 * pred_poses[0, 14, 0, 0]
        #zcom = (weight_position**2)*(2.0*pz1 - 2.0*pz2) * cos1 * pred_poses[0, 14, 0, 0]
        #yawcom = 4*(sin1*cos2 - cos1*sin2) * (- pred_poses[0, 15, 0, 0])

        #current pose
        px1 = px*0.0
        pz1 = pz*0.0
        cos1 = np.cos(ry*0.0)
        sin1 = np.sin(ry*0.0)
        
        xcom = (weight_position**2)*(2.0*px1 - 2.0*px2) * sin1 * pred_poses[0, 0, 0, 0]
        zcom = (weight_position**2)*(2.0*pz1 - 2.0*pz2) * cos1 * pred_poses[0, 0, 0, 0]
        yawcom = 4*(sin1*cos2 - cos1*sin2) * (- pred_poses[0, 1, 0, 0])
        
        #print(pred_poses[0, 14, 0, 0], pred_poses[0, 15, 0, 0], px1, pz1, cos1, sin1, px2, pz2, cos2, sin2)
        pose_reward_kyle = (xcom + zcom + 4.0*yawcom)/10.0 #9.0
        #print(pose_reward_kyle, xcom, zcom, yawcom, pred_poses[0, 14, 0, 0], pred_poses[0, 15, 0, 0], px1, pz1, cos1, sin1, px2, pz2, cos2, sin2)                        
        
        #print(Todom_listd.shape)
        #print(Tref_local.shape)
        #print(np.expand_dims(Tref_local.numpy(), axis=0).shape)
        #print(np.repeat(np.expand_dims(Tref_local.numpy(), axis=0), [step_length, 1, 1]).shape)

        image_loss = np.mean((Todom_listd - Tref_local.unsqueeze(0).repeat(step_length, 1, 1).numpy())**2)

        #print(TodomN.shape)
        #print(Tref_local.numpy().shape)
        pose_loss = np.mean((TodomN - Tref_local.numpy())**2)

        diff_pose_loss = np.mean((pred_poses[:, 0:2*(step_length-1)] - pred_poses[:, 2:2*step_length])**2)
        
        return -4.0*pose_loss, -0.4*image_loss, -0.5*diff_pose_loss, pose_reward_kyle
    """
    def rb_callback(self, msg):
        print("check topic reading!!")
        obs = tensor_dict_convert.from_ros_msg(msg)
        
        if False:
        #if obs["trans_mat"].sum() == self.prev_mat.sum():
            if (self.prev_t265[1, 3] - obs["t265_odom"][1, 3])**2 + (self.prev_t265[0, 3] - obs["t265_odom"][0, 3])**2 > 1:#jump detector
                self.flag_jump = 1
            self.buffer_loc.append(obs)
        else:   
            #print("check1")
            dict_info = {}
            for bl in range(len(self.buffer_loc)):
                #print("check2")
                Ttrans = self.prev_mat + (bl+1.0)*(obs["trans_mat"] - self.prev_mat)/len(self.buffer_loc)
                T265_cur = self.buffer_loc[bl]["t265_odom_cur"] 
                T265_next = self.buffer_loc[bl]["t265_odom"]     
                Tgoal_n = self.buffer_loc[bl]["goal_pose"]     
                
                #print(Tgoal_n)
                
                Traj_n = np.dot(Ttrans, T265_cur)
                Traj_next_n = np.dot(Ttrans, T265_next)
          
                axes_input = 'szyz'
                a1_traj, a2_traj, a3_traj = mat2euler(Traj_n[0:3, 0:3], axes=axes_input)

                #a3_traj = 155.0/180.0*3.1415

                Traj = np.zeros((4, 4))                                           
                Traj[0, 0] = np.cos(-a3_traj)
                Traj[0, 2] = np.sin(-a3_traj)
                Traj[1, 1] = 1.0
                Traj[2, 0] = -np.sin(-a3_traj)
                Traj[2, 2] = np.cos(-a3_traj)
                Traj[0, 3] = Traj_n[1, 3]
                Traj[2, 3] = -Traj_n[0, 3]
                Traj[3, 3] = 1.0
        
                a1_goal, a2_goal, a3_goal = mat2euler(Tgoal_n[0:3, 0:3], axes=axes_input)    
                
                #a3_goal = 5.0/180.0*3.1415
                    
                Tgoal = np.zeros((4, 4))                                           
                Tgoal[0, 0] = np.cos(-a3_goal)
                Tgoal[0, 2] = np.sin(-a3_goal)
                Tgoal[1, 1] = 1.0
                Tgoal[2, 0] = -np.sin(-a3_goal)
                Tgoal[2, 2] = np.cos(-a3_goal)
                Tgoal[0, 3] = Tgoal_n[1, 3]
                Tgoal[2, 3] = -Tgoal_n[0, 3]
                Tgoal[3, 3] = 1.0
                Tgoal_local = np.dot(np.linalg.inv(Traj), Tgoal)
                ang_diff = np.absolute(-a3_goal + a3_traj)      
        
                ang_diff = -a3_goal + a3_traj  
                ang_inv = np.arctan(Tgoal_local[0, 2]/Tgoal_local[0, 0])

                #print(ang_diff, ang_inv)                
        
                a1_traj_next, a2_traj_next, a3_traj_next = mat2euler(Traj_next_n[0:3, 0:3], axes=axes_input)        
                Ttraj_next = np.zeros((4, 4))                                           
                Ttraj_next[0, 0] = np.cos(-a3_traj_next)
                Ttraj_next[0, 2] = np.sin(-a3_traj_next)
                Ttraj_next[1, 1] = 1.0
                Ttraj_next[2, 0] = -np.sin(-a3_traj_next)
                Ttraj_next[2, 2] = np.cos(-a3_traj_next)
                Ttraj_next[0, 3] = Traj_next_n[1, 3]
                Ttraj_next[2, 3] = -Traj_next_n[0, 3]
                Ttraj_next[3, 3] = 1.0
                Tcur_next = np.dot(np.linalg.inv(Traj), Ttraj_next)                         
                
                Tnext_goal_local = np.dot(np.linalg.inv(Ttraj_next), Tgoal)
                
                dict_info["observations"] = self.buffer_loc[bl]["observations"].astype(np.uint8)
                #dict_info["next_observations"] = self.buffer_loc[bl]["next_observations"]
                dict_info["goal_observations"] = self.buffer_loc[bl]["goal_observations"].astype(np.uint8)        
                dict_info["actions"] = self.buffer_loc[bl]["actions"]
                dict_info["actions_previous"] = self.buffer_loc[bl]["actions_previous"]                
                dict_info["goal_pose"] = Tgoal_local
                dict_info["next_goal_pose"] = Tnext_goal_local
                dict_info["next_pose"] = Tcur_next             
                dict_info["vel_vec"] = self.buffer_loc[bl]["vel_vec"]
                flag_vw = self.buffer_loc[bl]["flag_vw"]
                dict_info["flag_col"]  = self.buffer_loc[bl]["flag_col"]
                
                if self.buffer_loc[bl]["acc_x"] > 5.0 or self.buffer_loc[bl]["acc_y"] > 5.0: 
                    dict_info["flag_acc"]  = 1.0
                else:
                    dict_info["flag_acc"]  = 0.0
                                                    
                dict_info["t265_odom"] = self.buffer_loc[bl]["t265_odom"]
                dict_info["t265_odom_cur"] = self.buffer_loc[bl]["t265_odom_cur"]                
                dict_info["goal_pose_before"] = self.buffer_loc[bl]["goal_pose"]      
                dict_info["trans"] = Ttrans
                dict_info["trans_mat"] = obs["trans_mat"] 
                dict_info["prev_mat"] = self.prev_mat  
                dict_info["rob_traj"] = self.buffer_loc[bl]["rob_traj"]

                #for pedestrian and robot traj
                rob_raw = self.buffer_loc[bl]["rob_traj"]
                ped_raw = self.buffer_loc[bl]["ped_traj"]

                Trans_next = np.zeros((4, 4)) 
                Trans_next[0, 0] = np.cos(-rob_raw[2,0])
                Trans_next[0, 2] = np.sin(-rob_raw[2,0])
                Trans_next[1, 1] = 1.0
                Trans_next[2, 0] = -np.sin(-rob_raw[2,0])
                Trans_next[2, 2] = np.cos(-rob_raw[2,0])
                Trans_next[0, 3] = -rob_raw[1,0]
                Trans_next[2, 3] = rob_raw[0,0]
                Trans_next[3, 3] = 1.0                
                Tinv_next = np.linalg.inv(Trans_next)
                
                Trans_current = np.zeros((4, 4)) 
                Trans_current[0, 0] = np.cos(-rob_raw[2,1])
                Trans_current[0, 2] = np.sin(-rob_raw[2,1])
                Trans_current[1, 1] = 1.0
                Trans_current[2, 0] = -np.sin(-rob_raw[2,1])
                Trans_current[2, 2] = np.cos(-rob_raw[2,1])
                Trans_current[0, 3] = -rob_raw[1,1]
                Trans_current[2, 3] = rob_raw[0,1]
                Trans_current[3, 3] = 1.0              
                Tinv_current = np.linalg.inv(Trans_current)
                                
                Tr_c_list = []
                Tr_n_list = []
                Tp_c_list = []
                Tp_n_list = []                                         
                for i in range(8):
                    Tr_n = np.zeros((4, 4)) 
                    Tr_n[0, 0] = np.cos(-rob_raw[2,i])
                    Tr_n[0, 2] = np.sin(-rob_raw[2,i])
                    Tr_n[1, 1] = 1.0
                    Tr_n[2, 0] = -np.sin(-rob_raw[2,i])
                    Tr_n[2, 2] = np.cos(-rob_raw[2,i])
                    Tr_n[0, 3] = -rob_raw[1,i]
                    Tr_n[2, 3] = rob_raw[0,i]
                    Tr_n[3, 3] = 1.0         
                      
                    Tr_c = np.zeros((4, 4)) 
                    Tr_c[0, 0] = np.cos(-rob_raw[2,i+1])
                    Tr_c[0, 2] = np.sin(-rob_raw[2,i+1])
                    Tr_c[1, 1] = 1.0
                    Tr_c[2, 0] = -np.sin(-rob_raw[2,i+1])
                    Tr_c[2, 2] = np.cos(-rob_raw[2,i+1])
                    Tr_c[0, 3] = -rob_raw[1,i+1]
                    Tr_c[2, 3] = rob_raw[0,i+1]
                    Tr_c[3, 3] = 1.0                             

                    Tp_n = np.zeros((4, 4)) 
                    Tp_n[0, 0] = np.cos(0.0)
                    Tp_n[0, 2] = np.sin(0.0)
                    Tp_n[1, 1] = 1.0
                    Tp_n[2, 0] = -np.sin(0.0)
                    Tp_n[2, 2] = np.cos(0.0)
                    Tp_n[0, 3] = ped_raw[0,i]
                    Tp_n[2, 3] = ped_raw[1,i]
                    Tp_n[3, 3] = 1.0         
                      
                    Tp_c = np.zeros((4, 4)) 
                    Tp_c[0, 0] = np.cos(0.0)
                    Tp_c[0, 2] = np.sin(0.0)
                    Tp_c[1, 1] = 1.0
                    Tp_c[2, 0] = -np.sin(0.0)
                    Tp_c[2, 2] = np.cos(0.0)
                    Tp_c[0, 3] = ped_raw[0,i+1]
                    Tp_c[2, 3] = ped_raw[1,i+1]
                    Tp_c[3, 3] = 1.0          
                    
                    if ped_raw[0,i] != 0.0 and ped_raw[1,i] != 0.0:    
                        Tp_n_list.append(np.expand_dims(np.matmul(Tinv_next, np.matmul(Tr_n, Tp_n)), axis=0))
                    else:
                        Tp_n_list.append(np.expand_dims(np.eye(4), axis=0))
                    if ped_raw[0,i+1] != 0.0 and ped_raw[1,i+1] != 0.0:
                        Tp_c_list.append(np.expand_dims(np.matmul(Tinv_current, np.matmul(Tr_c, Tp_c)), axis=0))
                    else:
                        Tp_c_list.append(np.expand_dims(np.eye(4), axis=0))                        
                    
                    Tr_n_list.append(np.expand_dims(np.matmul(Tinv_next, Tr_n), axis=0))
                    Tr_c_list.append(np.expand_dims(np.matmul(Tinv_current, Tr_c), axis=0))
                
                Tp_n_listcat = np.concatenate(Tp_n_list, axis=0)
                Tp_c_listcat = np.concatenate(Tp_c_list, axis=0)
                Tr_n_listcat = np.concatenate(Tr_n_list, axis=0)
                Tr_c_listcat = np.concatenate(Tr_c_list, axis=0)                                                
                
                ped_n_xy = np.concatenate((Tp_n_listcat[:,0,3], Tp_n_listcat[:,2,3]), axis=0)
                ped_c_xy = np.concatenate((Tp_c_listcat[:,0,3], Tp_c_listcat[:,2,3]), axis=0)   
                robot_n_xy = np.concatenate((Tr_n_listcat[:,0,3], Tr_n_listcat[:,2,3]), axis=0)
                robot_c_xy = np.concatenate((Tr_c_listcat[:,0,3], Tr_c_listcat[:,2,3]), axis=0)

                for i in range(6):
                    if ped_n_xy[i+1] == 0 and ped_n_xy[i+1+8] == 0:
                        if ped_n_xy[i] != 0 and ped_n_xy[i+2] != 0:
                            ped_n_xy[i+1] = (ped_n_xy[i] + ped_n_xy[i+2])*0.5
                            ped_n_xy[i+1+8] = (ped_n_xy[i+8] + ped_n_xy[i+2+8])*0.5
                            #print("hokan A")                            
                    if ped_n_xy[i+1] != 0 and ped_n_xy[i+1+8] != 0:
                        if ped_n_xy[i] != 0 and ped_n_xy[i+2] != 0:
                            ped_n_xy[i+1] = (ped_n_xy[i] + ped_n_xy[i+1] + ped_n_xy[i+2])*0.3333
                            ped_n_xy[i+1+8] = (ped_n_xy[i+8] + ped_n_xy[i+1+8] + ped_n_xy[i+2+8])*0.3333
                            
                    if ped_c_xy[i+1] == 0 and ped_c_xy[i+1+8] == 0:
                        if ped_c_xy[i] != 0 and ped_c_xy[i+2] != 0:
                            ped_c_xy[i+1] = (ped_c_xy[i] + ped_c_xy[i+2])*0.5
                            ped_c_xy[i+1+8] = (ped_c_xy[i+8] + ped_c_xy[i+2+8])*0.5
                            #print("hokan A")                            
                    if ped_c_xy[i+1] != 0 and ped_c_xy[i+1+8] != 0:
                        if ped_c_xy[i] != 0 and ped_c_xy[i+2] != 0:
                            ped_c_xy[i+1] = (ped_c_xy[i] + ped_c_xy[i+1] + ped_c_xy[i+2])*0.3333
                            ped_c_xy[i+1+8] = (ped_c_xy[i+8] + ped_c_xy[i+1+8] + ped_c_xy[i+2+8])*0.3333                            
                            #print("hokan B")
                            
                if ped_n_xy[0] == 0 and ped_n_xy[8] == 0:
                    if ped_n_xy[1] != 0 and ped_n_xy[2] != 0:
                        delta_x = ped_n_xy[1] - ped_n_xy[2]
                        delta_z = ped_n_xy[9] - ped_n_xy[10]       
                        ped_n_xy[0] = delta_x + ped_n_xy[1]
                        ped_n_xy[8] = delta_z + ped_n_xy[9]
                        #print("hokan C")
                                  
                if ped_n_xy[7] == 0 and ped_n_xy[15] == 0:
                    if ped_n_xy[6] != 0 and ped_n_xy[5] != 0:                
                        delta_x = ped_n_xy[6] - ped_n_xy[5]
                        delta_z = ped_n_xy[14] - ped_n_xy[13]       
                        ped_n_xy[0] = delta_x + ped_n_xy[6]
                        ped_n_xy[15] = delta_z + ped_n_xy[14]     
                        #print("hokan D")
                        
                if ped_c_xy[0] == 0 and ped_c_xy[8] == 0:
                    if ped_c_xy[1] != 0 and ped_c_xy[2] != 0:
                        delta_x = ped_c_xy[1] - ped_c_xy[2]
                        delta_z = ped_c_xy[9] - ped_c_xy[10]       
                        ped_c_xy[0] = delta_x + ped_c_xy[1]
                        ped_c_xy[8] = delta_z + ped_c_xy[9]
                        #print("hokan C")
                                  
                if ped_c_xy[7] == 0 and ped_c_xy[15] == 0:
                    if ped_c_xy[6] != 0 and ped_c_xy[5] != 0:                
                        delta_x = ped_c_xy[6] - ped_c_xy[5]
                        delta_z = ped_c_xy[14] - ped_c_xy[13]       
                        ped_c_xy[0] = delta_x + ped_c_xy[6]
                        ped_c_xy[15] = delta_z + ped_c_xy[14]                                       
                        #print("hokan D")
                                                                 
                #print(ped_n_xy.shape, ped_c_xy.shape, robot_n_xy.shape, robot_c_xy.shape)
                #print("check!!")

                dict_info["robot_next"] = robot_n_xy
                dict_info["robot_current"] = robot_c_xy                
                dict_info["ped_next"] = ped_n_xy
                dict_info["ped_current"] = ped_c_xy                

                dict_info["pc_cur"] = self.buffer_loc[bl]["pc_cur"]  
                dict_info["pc_next"] = self.buffer_loc[bl]["pc_next"] 

                """
                #print("before", dict_info["next_pose"])
                Tnext = torch.from_numpy(dict_info["next_pose"].copy()) 
                px_next = Tnext[0, 3]
                pz_next = Tnext[2, 3]                                                
                Tnext[0, 3] = 2.0*torch.from_numpy(dict_info["next_pose"].copy())[0, 3]
                Tnext[2, 3] = 2.0*torch.from_numpy(dict_info["next_pose"].copy())[2, 3] 
                #print("after", dict_info["next_pose"])
                
                Tref = torch.from_numpy(dict_info["goal_pose"].copy())
                px_ref = Tref[0, 3]
                pz_ref = Tref[2, 3]                 
                #Tref[0, 3] = 2.0*torch.from_numpy(dict_info["next_goal_pose"].copy())[0, 3]
                #Tref[2, 3] = 2.0*torch.from_numpy(dict_info["next_goal_pose"].copy())[2, 3]
                Tref[0, 3] = 2.0*torch.from_numpy(dict_info["goal_pose"].copy())[0, 3]
                Tref[2, 3] = 2.0*torch.from_numpy(dict_info["goal_pose"].copy())[2, 3]
                pose_loss, image_loss, diff_pose_loss, pose_kyle = self.func_pose_reward(dict_info["vel_vec"], Tref)
                        
                pose_reward = 4.0 * torch.mean((Tref - Tnext)**2, dim=(0,1))/8.0 
                  
                goal_eval = torch.sqrt((px_next - px_ref)**2 + (pz_next - pz_ref)**2) < 0.5*torch.ones(1)
                goal_over = Tref[2, 3] < 0.0

                goal_reward = -3.0*goal_eval.float()# + 10.0*goal_over.float()     

                pc_hom = torch.ones((1, 128, 416))
                transform_l = T.Resize((128,832)) 

                pc_cur_rl = transform_l(torch.from_numpy(dict_info["pc_next"]))
                pcs_cur_f_rl = pc_cur_rl[:,:,0:416]
                pcs_cur_b_rl = pc_cur_rl[:,:,416:832] 
                                                
                bias = 10                                 
                points_next_f = pcs_cur_f_rl[0:3,64-2*bias:64+2*bias,bias:416-bias].reshape(3, -1)
                points_next_b = pcs_cur_b_rl[0:3,64-2*bias:64+2*bias,bias:416-bias].reshape(3, -1)
                points_next = torch.cat((points_next_f, points_next_b), axis=1)
                
                robot_size = 0.5*torch.ones(1, 1, 1)
                #col_eval = points_next[0,:]**2 + points_next[1,:]**2 + points_next[2,:]**2 < (roboTgoal_localt_size.squeeze(2).squeeze(1))**2
                #col_eval_count = torch.sum(col_eval.float(), axis=0) > 30*torch.ones(1) #threshold 30 points       
                #col_reward = col_eval_count.float()     
                
                if dict_info["flag_col"] > 0.5 or dict_info["flag_acc"] > 0.5:
                    #col_reward = 3.0*dict_info["flag_col"]
                    col_reward = torch.tensor(3.0)
                else:
                    col_reward = torch.tensor(0.0)
                                    
                #print(col_reward, (points_next[0,:]**2 + points_next[1,:]**2 + points_next[2,:]**2).min(), (points_next[0,:]**2 + points_next[1,:]**2 + points_next[2,:]**2).max(), (points_next[0,:]**2 + points_next[1,:]**2 + points_next[2,:]**2).mean())
                
                ped_past_cat_next = torch.clamp(torch.from_numpy(dict_info["ped_next"]), min=-10.0, max=10.0)
                       
                ped_dist = torch.sqrt((ped_past_cat_next[0])**2 + (ped_past_cat_next[8])**2)                       
                ps_eval = ped_dist < (robot_size.squeeze(1).squeeze(1) + 0.5)

                if ped_past_cat_next[0] == 0.0 and ped_past_cat_next[8] == 0.0:
                    ps_eval = torch.tensor(0.0)
                ps_reward = 1.0*ps_eval.float()

                with open(newpath + '/reward.txt', 'a') as f:
                        f.write("pose" + " " + str(pose_reward.item()) + " " + "goal" + " " + str(goal_reward.item()) + " " + "col" + " " + str(col_reward.item()) + " " + "col_count" + " " + str(col_reward) + " " + "ps" + " " + str(ps_reward.item()) + " " + "ps_dist" + " " + str(ped_dist.item()) + " " + "count" + " " + str(self.counter) + " " + "pose_loss" + " " + str(pose_loss) + " " + "image_loss" + " " + str(image_loss) + " " + "diff_loss" + " " + str(diff_pose_loss) + " " + "pose_kype" + " " + str(pose_kyle.item()) + " " + "\n")                             
                                                    
                remove_znegative = Tgoal_local[2, 3] < 0.0
                remove_distance = Tgoal_local[2, 3]**2 + Tgoal_local[0, 3]**2 > 2.0**2 #1.8                  
                #remove_angle = np.absolute(np.arctan(Tgoal_local[0, 2]/Tgoal_local[0, 0])) > 1.55
                
                theta_thres = 2.0 #1.6
                remove_angle = ang_diff > theta_thres or ang_diff < - theta_thres
                remove_data = np.array(remove_distance + remove_angle, dtype=bool) #remove_znegative
                
                Ttraj = np.dot(Ttrans, T265_cur)
                with open(newpath + '/data_check.txt', 'a') as f:
                    f.write(str(bl) + " " + str(len(self.buffer_loc)) + " " + str(self.buffer_loc[bl]["counter_trans"]) + " " + str(obs["total_count"]) + " " + str(Ttraj[0, 3]) + " " + str(Ttraj[1, 3]) + " " + str(Ttraj[2, 3]) + " " + str(Ttraj[0, 2]/Ttraj[0, 0]) + " " + str(Tgoal_local[0, 3]) + " " + str(Tgoal_local[1, 3])+ " " + str(Tgoal_local[2, 3]) + " " + str(Tgoal_local[0, 2]/Tgoal_local[0, 0]) + " " + str(Tnext_goal_local[0, 3]) + " " + str(Tnext_goal_local[1, 3]) + " " + str(Tnext_goal_local[2, 3]) + " " + str(Tnext_goal_local[0, 2]/Tnext_goal_local[0, 0]) + " " + str(Tcur_next[0, 3]) + " " + str(Tcur_next[1, 3]) + " " + str(Tcur_next[2, 3]) + " " + str(Tcur_next[0, 2]/Tcur_next[0, 0]) + " " + str(Tgoal[0, 3]) + " " + str(Tgoal[1, 3]) + " " + str(Tgoal[2, 3]) + " " + str(Tgoal[0, 2]/Tgoal[0, 0]) + " " + str(T265_cur[0, 3]) + " " + str(T265_cur[1, 3]) + " " + str(T265_cur[2, 3]) + " " + str(T265_cur[0, 2]/T265_cur[0, 0]) + " " + str(self.flag_jump) + " " + str(remove_distance) + " " + str(remove_angle) + " " + str(ang_diff) + "\n")           
                """
                """
                dict_info_save = dict_info.copy()
                with open(newpath + "/" + str(self.counter) + '_saved_dictionary.pkl', 'wb') as f:
                    observations = np.copy(dict_info["observations"])
                    goal_observations = np.copy(dict_info["goal_observations"])
                
                    #to convert the image from float to uint8 to compress the pickle file 
                    dict_info_save["observations"] = observations.astype(np.uint8)
                    dict_info_save["goal_observations"] = goal_observations.astype(np.uint8)            
                    #dict_info_save["observations"] = (127.5*observations + 127.5).astype(np.uint8)
                    #dict_info_save["goal_observations"] = (127.5*goal_observations + 127.5).astype(np.uint8)                   
                
                    pickle.dump(dict_info_save, f)
                    
                self.counter += 1 
                """
                remove_znegative = Tgoal_local[2, 3] < 0.0
                remove_distance = Tgoal_local[2, 3]**2 + Tgoal_local[0, 3]**2 > 2.0**2 #1.8                  
                #remove_angle = np.absolute(np.arctan(Tgoal_local[0, 2]/Tgoal_local[0, 0])) > 1.55
                
                theta_thres = 2.0 #1.6
                remove_angle = ang_diff > theta_thres or ang_diff < - theta_thres
                remove_data = np.array(remove_distance + remove_angle, dtype=bool) #remove_znegative
                                
                """
                T265_next = np.dot(self.rotmat_tf, self.buffer_loc[bl]["t265_odom"])     #T265->fisheye cam.
                T265_cur = np.dot(self.rotmat_tf, self.buffer_loc[bl]["t265_odom_cur"])  #T265->fisheye cam
                T_goal = np.dot(self.rotmat_tf, self.buffer_loc[bl]["goal_pose"])        #T265->fisheye cam           
                Tcur_next = np.dot(np.linalg.inv(T265_cur), T265_next)
                
                T265_p = self.buffer_loc[bl]["t265_odom"]
                #Ttrans = self.prev_mat + self.buffer_loc[bl]["counter_trans"]*(obs["trans_mat"] - self.prev_mat)/obs["total_count"]Tinv_current
                Ttrans = self.prev_mat + (bl+1.0)*(obs["trans_mat"] - self.prev_mat)/len(self.buffer_loc)
                
                #print("check counter", self.buffer_loc[bl]["counter_trans"], obs["total_count"])
                Tgoal_local = np.dot(np.linalg.inv(np.dot(Ttrans, T265_cur)), T_goal)
                
                dict_info["observations"] = self.buffer_loc[bl]["observations"]
                dict_info["next_observations"] = self.buffer_loc[bl]["next_observations"]
                dict_info["actions"] = self.buffer_loc[bl]["actions"]
                dict_info["goal_pose"] = Tgoal_local
                Tnext_goal_local = np.dot(np.linalg.inv(Tcur_next), Tgoal_local) 
                dict_info["next_goal_pose"] = Tnext_goal_localdict_info
                dict_info["next_pose"] = Tcur_next 
                
                dict_info["t265_odom"] = self.buffer_loc[bl]["t265_odom"]
                dict_info["t265_odom_cur"] = self.buffer_loc[bl]["t265_odom_cur"]                
                dict_info["goal_pose_before"] = self.buffer_loc[bl]["goal_pose"]      
                dict_info["trans"] = Ttrans
                dict_info["trans_mat"] = obs["trans_mat"] 
                dict_info["prev_mat"] = self.prev_mat                
                """
                #print(self.flag_jump, self.flag_init, remove_data, remove_znegative, remove_distance, remove_angle)
                #remove_data = False
                
                if remove_data == True or self.flag_jump != 0:
                #if True:
                    ifs = random.randint(0, 12+1)
                    blg = min(bl+ifs, len(self.buffer_loc)-1)
                    
                    dict_info["goal_observations"] = self.buffer_loc[blg]["observations"][:,6:12]
                    rob_raw_blg = self.buffer_loc[blg]["rob_traj"]

                    Trans_curg = np.zeros((4, 4)) 
                    Trans_curg[0, 0] = np.cos(-rob_raw_blg[2,1])
                    Trans_curg[0, 2] = np.sin(-rob_raw_blg[2,1])
                    Trans_curg[1, 1] = 1.0
                    Trans_curg[2, 0] = -np.sin(-rob_raw_blg[2,1])
                    Trans_curg[2, 2] = np.cos(-rob_raw_blg[2,1])
                    Trans_curg[0, 3] = -rob_raw_blg[1,1]
                    Trans_curg[2, 3] = rob_raw_blg[0,1]
                    Trans_curg[3, 3] = 1.0              
                    
                    #goal_local 
                    dict_info["goal_pose"] = np.matmul(Tinv_current, Trans_curg)
                    dict_info["next_goal_pose"] = np.matmul(Tinv_next, Trans_curg)                    

                print("check save1", newpath)
                dict_info_save = dict_info.copy()
                with open(newpath + "/" + str(self.counter) + '_saved_dictionary.pkl', 'wb') as f:
                    print("check save2")
                    observations = np.copy(dict_info["observations"])
                    goal_observations = np.copy(dict_info["goal_observations"])
                
                    dict_info_save["observations"] = observations.astype(np.uint8)
                    dict_info_save["goal_observations"] = goal_observations.astype(np.uint8)                            
                
                    pickle.dump(dict_info_save, f)
                    
                self.counter += 1 


                #if True:          
                #if self.flag_jump == 0 and self.flag_init == 1 and not remove_data:#flag_vw > 0.5 and
                
                
                  
                """
                if self.flag_init == 1:              
                    #with open('/media/noriaki/Noriaki_Data2/save_data_RL/' + str(self.counter) + '_saved_dictionary.pkl', 'wb') as f:
                    #with open(newpath + "/" + str(self.counter) + '_saved_dictionary_after.pkl', 'wb') as f:                    
                    #    pickle.dump(dict_info, f)
                
                    Ttraj = np.dot(Ttrans, T265_cur)
                    with open(newpath + '/data.txt', 'a') as f:
                        f.write(str(bl) + " " + str(len(self.buffer_loc)) + " " + str(self.buffer_loc[bl]["counter_trans"]) + " " + str(obs["total_count"]) + " " + str(Ttraj[0, 3]) + " " + str(Ttraj[1, 3]) + " " + str(Ttraj[2, 3]) + " " + str(Ttraj[0, 2]/Ttraj[0, 0]) + " " + str(Tgoal_local[0, 3]) + " " + str(Tgoal_local[1, 3])+ " " + str(Tgoal_local[2, 3]) + " " + str(Tgoal_local[0, 2]/Tgoal_local[0, 0]) + " " + str(Tnext_goal_local[0, 3]) + " " + str(Tnext_goal_local[1, 3]) + " " + str(Tnext_goal_local[2, 3]) + " " + str(Tnext_goal_local[0, 2]/Tnext_goal_local[0, 0]) + " " + str(Tcur_next[0, 3]) + " " + str(Tcur_next[1, 3]) + " " + str(Tcur_next[2, 3]) + " " + str(Tcur_next[0, 2]/Tcur_next[0, 0]) + " " + str(Tgoal[0, 3]) + " " + str(Tgoal[1, 3]) + " " + str(Tgoal[2, 3]) + " " + str(Tgoal[0, 2]/Tgoal[0, 0]) + " " + str(T265_cur[0, 3]) + " " + str(T265_cur[1, 3]) + " " + str(T265_cur[2, 3]) + " " + str(T265_cur[0, 2]/T265_cur[0, 0]) + " " + str(self.flag_jump) + "\n")                             
                    self.rb_queue.put(dict_info)   
                    #self.counter += 1               
                """
            #reset buffer
            self.buffer_loc = []
            self.buffer_loc.append(obs) 
            self.prev_mat = obs["trans_mat"]
            self.flag_jump = 0
            self.flag_init = 1
        
        self.prev_t265 = obs["t265_odom"]                    
        #self.rb_queue.put(obs)                        

    """
    def param_pub_callback(self, _):
        params = None
        while True:
            try:
                params = self.param_queue.get_nowait()
                #print(params)
            except queue.Empty:
                break
        if params is not None:
            print("sent parameters!!")
            self.param_publisher.publish(
                tensor_dict_convert.to_ros_msg(params)
            )
    """

"""
class PreprocessedDataset(chainer.dataset.DatasetMixin):

    def __init__(self, path, root, random=True):
        self.base = chainer.datasets.LabeledImageDataset(path, root)
        self.random = random

    def __len__(self):
        return len(self.base)

    def get_example(self, i):
        image, label = self.base[i]
        _, h, w = image.shape
        imagex = (image - 127.5)/127.5

        return imagex, label

class Vmpc(chainer.Chain):
    def __init__(self, in_ch):
        layers = {}
        w = chainer.initializers.Normal(0.02)
        layers['c0'] = L.Convolution2D(in_ch, 64, 3, 1, 1, initialW=w)
        layers['c1'] = CBR(64, 128, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c2'] = CBR(128, 256, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c3'] = CBR(256, 512, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c4'] = CBR(512, 512, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c5'] = CBR(512, 512, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c6'] = CBR(512, 512, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c7'] = L.Convolution2D(512, 16, 4, 2, 1, initialW=w)
        super(Vmpc, self).__init__(**layers)

    def __call__(self, x):
        hs = F.leaky_relu(self.c0(x))
        for i in range(1,8):
            hs = self['c%d'%i](hs)
        return hs

class CBR(chainer.Chain):
    def __init__(self, ch0, ch1, bn=True, sample='down', activation=F.relu, dropout=False):
        self.bn = bn
        self.activation = activation
        self.dropout = dropout
        layers = {}
        w = chainer.initializers.Normal(0.02)
        if sample=='down':
            layers['c'] = L.Convolution2D(ch0, ch1, 4, 2, 1, initialW=w)
        else:
            layers['c'] = L.Deconvolution2D(ch0, ch1, 4, 2, 1, initialW=w)
        if bn:
            layers['batchnorm'] = L.BatchNormalization(ch1)
        super(CBR, self).__init__(**layers)
        
    def __call__(self, x):
        h = self.c(x)
        if self.bn:
            h = self.batchnorm(h)
        if self.dropout:
            h = F.dropout(h)
        if not self.activation is None:
            h = self.activation(h)
        return h

class Encoder(chainer.Chain):
    def __init__(self, in_ch):
        layers = {}
        w = chainer.initializers.Normal(0.02)
        layers['c0'] = L.Convolution2D(in_ch, 64, 3, 1, 1, initialW=w)
        layers['c1'] = CBR(64, 128, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c2'] = CBR(128, 256, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c3'] = CBR(256, 512, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c4'] = CBR(512, 512, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c5'] = CBR(512, 512, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c6'] = CBR(512, 512, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c7'] = CBR(512, 512, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        super(Encoder, self).__init__(**layers)

    def __call__(self, x):
        hs = [F.leaky_relu(self.c0(x))]
        for i in range(1,8):
            hs.append(self['c%d'%i](hs[i-1]))
        return hs

class Decoder(chainer.Chain):
    def __init__(self, out_ch):
        layers = {}
        w = chainer.initializers.Normal(0.02)
        layers['c0'] = CBR(528, 512, bn=True, sample='up', activation=F.relu, dropout=False)
        layers['c1'] = CBR(512, 512, bn=True, sample='up', activation=F.relu, dropout=False)
        layers['c2'] = CBR(512, 512, bn=True, sample='up', activation=F.relu, dropout=False)
        layers['c3'] = CBR(512, 512, bn=True, sample='up', activation=F.relu, dropout=False)
        layers['c4'] = CBR(512, 256, bn=True, sample='up', activation=F.relu, dropout=False)
        layers['c5'] = CBR(256, 128, bn=True, sample='up', activation=F.relu, dropout=False)
        layers['c6'] = CBR(128, 64, bn=True, sample='up', activation=F.relu, dropout=False)
        layers['c7'] = L.Convolution2D(64, out_ch, 3, 1, 1, initialW=w)
        super(Decoder, self).__init__(**layers)

    def __call__(self, hs, vres, wres):
        z = F.concat((hs[-1],vres,wres), axis=1)
        h = self.c0(z)
        for i in range(1,8):
            if i<7:
                h = self['c%d'%i](h)
            else:
                h = F.tanh(self.c7(h))
        return h

class Generator(chainer.Chain):
    def __init__(self, wscale=0.02):
        initializer = chainer.initializers.Normal(wscale)
        super(Generator, self).__init__(
            l0z = L.Linear(nz, 8*8*512, initialW=initializer),
            dc1 = L.Deconvolution2D(512, 256, 4, stride=2, pad=1, initialW=initializer),
            dc2 = L.Deconvolution2D(256, 128, 4, stride=2, pad=1, initialW=initializer),
            dc3 = L.Deconvolution2D(128, 64, 4, stride=2, pad=1, initialW=initializer),
            dc4 = L.Deconvolution2D(64, 3, 4, stride=2, pad=1, initialW=initializer),
            bn0l = L.BatchNormalization(8*8*512),
            bn0 = L.BatchNormalization(512),
            bn1 = L.BatchNormalization(256),
            bn2 = L.BatchNormalization(128),
            bn3 = L.BatchNormalization(64),
        )
        
    def __call__(self, z, test=False):
        h = F.reshape(F.relu(self.bn0l(self.l0z(z))), (z.data.shape[0], 512, 8, 8))
        h = F.relu(self.bn1(self.dc1(h)))
        h = F.relu(self.bn2(self.dc2(h)))
        h = F.relu(self.bn3(self.dc3(h)))
        x = (self.dc4(h))
        return x

class invG(chainer.Chain):
    def __init__(self, wscale=0.02):
        initializer = chainer.initializers.Normal(wscale)
        super(invG, self).__init__(
            c0 = L.Convolution2D(3, 64, 4, stride=2, pad=1, initialW=initializer),
            c1 = L.Convolution2D(64, 128, 4, stride=2, pad=1, initialW=initializer),
            c2 = L.Convolution2D(128, 256, 4, stride=2, pad=1, initialW=initializer),
            c3 = L.Convolution2D(256, 512, 4, stride=2, pad=1, initialW=initializer),
            l4l = L.Linear(8*8*512, nz, initialW=initializer),
            bn0 = L.BatchNormalization(64),
            bn1 = L.BatchNormalization(128),
            bn2 = L.BatchNormalization(256),
            bn3 = L.BatchNormalization(512),
        )
        
    def __call__(self, x, test=False):
        h = F.relu(self.c0(x))
        h = F.relu(self.bn1(self.c1(h)))
        h = F.relu(self.bn2(self.c2(h))) 
        h = F.relu(self.bn3(self.c3(h)))
        l = self.l4l(h)
        return l
"""
def clip_img(x):
	return np.float32(-1 if x<-1 else (1 if x>1 else x))
"""
class ELU(function.Function):

    def __init__(self, alpha=1.0):
        self.alpha = numpy.float32(alpha)

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type, = in_types

        type_check.expect(
            x_type.dtype == numpy.float32,
        )

    def forward_cpu(self, x):
        y = x[0].copy()
        neg_indices = x[0] < 0
        y[neg_indices] = self.alpha * (numpy.exp(y[neg_indices]) - 1)
        return y,

    def forward_gpu(self, x):
        y = cuda.elementwise(
            'T x, T alpha', 'T y',
            'y = x >= 0 ? x : alpha * (exp(x) - 1)', 'elu_fwd')(
                x[0], self.alpha)
        return y,

    def backward_cpu(self, x, gy):
        gx = gy[0].copy()
        neg_indices = x[0] < 0
        gx[neg_indices] *= self.alpha * numpy.exp(x[0][neg_indices])
        return gx,

    def backward_gpu(self, x, gy):
        gx = cuda.elementwise(
            'T x, T gy, T alpha', 'T gx',
            'gx = x >= 0 ? gy : gy * alpha * exp(x)', 'elu_bwd')(
                x[0], gy[0], self.alpha)
        return gx,

def elu(x, alpha=1.0):
    # Exponential Linear Unit function.
    # https://github.com/muupan/chainer-elu
    return ELU(alpha=alpha)(x)

class Discriminator(chainer.Chain):
    def __init__(self, wscale=0.02):
        initializer = chainer.initializers.Normal(wscale)
        super(Discriminator, self).__init__(
            c0 = L.Convolution2D(3, 64, 4, stride=2, pad=1, initialW=initializer),
            c1 = L.Convolution2D(64, 128, 4, stride=2, pad=1, initialW=initializer),
            c2 = L.Convolution2D(128, 256, 4, stride=2, pad=1, initialW=initializer),
            c3 = L.Convolution2D(256, 512, 4, stride=2, pad=1, initialW=initializer),
            l4l = L.Linear(8*8*512, 2, initialW=initializer),
            bn0 = L.BatchNormalization(64),
            bn1 = L.BatchNormalization(128),
            bn2 = L.BatchNormalization(256),
            bn3 = L.BatchNormalization(512),
        )
        
    def __call__(self, x, test=False):
        h = elu(self.c0(x))
        h = elu(self.bn1(self.c1(h)))
        h = elu(self.bn2(self.c2(h))) 
        h = elu(self.bn3(self.c3(h)))
        l = self.l4l(h)
        return h

class FL(chainer.Chain):
    def __init__(self, wscale=0.02):
        initializer = chainer.initializers.Normal(wscale)
        super(FL, self).__init__(
            l_img = L.Linear(3*128*128, 1, initialW=initializer),
            l_dis = L.Linear(512*8*8, 1, initialW=initializer),
            l_fdis = L.Linear(512*8*8, 1, initialW=initializer),
            l_FL = L.Linear(3, 1, initialW=initializer),
        )
        
    def __call__(self, img_error, dis_error, dis_output, test=False):
        h = F.reshape(F.absolute(img_error), (img_error.data.shape[0], 3*128*128))
        h = self.l_img(h)
        g = F.reshape(F.absolute(dis_error), (dis_error.data.shape[0], 512*8*8))
        g = self.l_dis(g)
        f = F.reshape(dis_output, (dis_output.data.shape[0], 512*8*8))
        f = self.l_fdis(f)
        ghf = F.sigmoid(self.l_FL(F.concat((h,g,f), axis=1)))
        return ghf
"""
def preprocess_image(msg):
    cv_img = bridge.imgmsg_to_cv2(msg)
    #print("cv_img", cv_img.shape)
    cv_resize_n = cv2.resize(cv_img[:, 0:560], (rsizex_lnp, rsizey_lnp), cv2.INTER_AREA)
    #print(cv_resize_n.shape)  
    cv_resizex = cv_resize_n.transpose(2, 0, 1)
    in_imgcc1 = np.array([cv_resizex], dtype=np.float32)
    in_img1 = (in_imgcc1 - 127.5)/127.5

    #img_nn_cL = mask_brrc * (in_img1 + 1.0) -1.0 #mask
    img_nn_cL = 2.0*(in_img1 - 0.5)
    img = img_nn_cL.astype(np.float32)

    return img, cv_resize_n

def preprocess_image_gen(msg):
    cv_img = bridge.imgmsg_to_cv2(msg, "bgr8")
    cv_resize_n = cv2.resize(cv_img, (rsizex, rsizey), cv2.INTER_AREA)
    cv_resizex = cv_resize_n.transpose(2, 0, 1)
    in_imgcc1 = np.array([cv_resizex], dtype=np.float32)
    in_img1 = (in_imgcc1 - 127.5)/127.5

    img_nn_cL = (in_img1 + 1.0) -1.0 #mask
    img = img_nn_cL.astype(np.float32)

    return img

def imgmsg_to_numpy_realsense(msg):
    cv_resize_n = bridge.imgmsg_to_cv2(msg)
    cv_resizex = cv_resize_n.transpose(2, 0, 1)
    in_imgcc1 = np.array([cv_resizex], dtype=np.float32)
    in_img1 = (in_imgcc1 - 127.5)/127.5

    img_nn_cL = (in_img1 + 1.0) -1.0
    img = img_nn_cL.astype(np.float32)

    return img

def imgmsg_to_numpy(msg):
    cv_resize_n = bridge.imgmsg_to_cv2(msg)
    cv_resizex = cv_resize_n.transpose(2, 0, 1)
    in_imgcc1 = np.array([cv_resizex], dtype=np.float32)
    in_img1 = (in_imgcc1 - 127.5)/127.5

    img_nn_cL = mask_brrc * (in_img1 + 1.0) -1.0 #mask
    img = img_nn_cL.astype(np.float32)

    return img

def get_rotation(msg):
    global roll, pitch, yaw
    global x_pos, y_pos, z_pos
    global topic_odom
    topic_odom = msg

    orientation_q = msg.pose.pose.orientation
    position = msg.pose.pose.position
    x_pos = position.x
    y_pos = position.y
    z_pos = position.z
    orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
    (roll, pitch, yaw) = euler_from_quaternion (orientation_list)

def sinc_apx(angle):
    return torch.sin(3.141592*angle + 0.000000001)/(3.141592*angle + 0.000000001)

#from Gibson
def xyz2mat(xyz):
	trans_mat = np.eye(4)
	trans_mat[-1, :3] = xyz
	return trans_mat

def quat2rotmat(quat):
	"""
	:param quat: quaternion in w,x,y,z
	:return: rotation matrix 4x4
	"""
	rot_mat = np.eye(4)
	rot_mat[:3, :3] = quaternions.quat2mat(quat)
	return rot_mat

def mat2xyz(mat):
	xyz = mat[-1, :3]
	xyz[np.isnan(xyz)] = 0
	return xyz

def safemat2quat(mat):
	"""
	:param mat: 4x4 matrix
	:return: quaternion in w,x,y,z
	"""
	quat = np.array([1, 0, 0, 0])
	try:
		quat = mat2quat(mat)
	except:
		pass
	quat[np.isnan(quat)] = 0
	return quat

def catRT(R, T):
	return np.concatenate((np.concatenate((R, T), axis=1), np.array([[0.0, 0.0, 0.0, 1.0]])), axis=0)


def xyzw2wxyz(orn):
	"""
	:param orn: quaternion in xyzw
	:return: quaternion in wxyz
	"""
	return [orn[-1], orn[0], orn[1], orn[2]]

def robot_pos_model_fix(linear_vel, angular_vel):
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
        Todom[:, 0, 3] = x #weighting for position
        Todom[:, 2, 3] = z #weighting for position
        Todom[:, 3, 3] = 1.0        
        
        #Tacc = Tacc @ Todom
        Tacc = torch.matmul(Tacc, Todom)
        
        #Taccd = Tacc.clone()
        #Taccd[:, 0, 3] = 2.0*Tacc.clone()[:, 0, 3]
        #Taccd[:, 2, 3] = 2.0*Tacc.clone()[:, 2, 3]        
        pyaw.append(torch.arctan(Tacc[:, 0, 2]/(Tacc[:, 0, 0] + 0.000000001)))        
        px.append(Tacc[:, 0, 3])
        pz.append(Tacc[:, 2, 3])        
    return px, pz, pyaw

def callback_rsense(msg_1):
    global i
    global j
    #global timagev
    #global Nline
    #global count
    #global Lpast
    #global inputgpu
    #global prefv
    #global opt_biasv
    #global swopt
    #global Lmin
    global vwkeep
    global topic_velcmd
    #global Lth
    #global mask_brrc
    #global imgbt, imgrt, imggt   
    #global prev_cmd_vel 
    #global goal_img

    j = j + 1
    #print(j, goal_img.shape)
    #print j
    if j == 1:
        cur_img = preprocess_image_gen(msg_1) #current image
        
        #cur_img_360 = cur_img
        #goal_img_360 = goal_img

        #standard deviation and mean for current image
        imgbc = (np.reshape(cur_img[0][0],(1,128,128)) + 1.0)*0.5
        imggc = (np.reshape(cur_img[0][1],(1,128,128)) + 1.0)*0.5
        imgrc = (np.reshape(cur_img[0][2],(1,128,128)) + 1.0)*0.5
        mean_cbgr = np.zeros((3,1))
        std_cbgr = np.zeros((3,1))
        mean_ct = np.zeros((3,1))
        std_ct = np.zeros((3,1))
        mean_cbgr[0] = np.sum(imgbc)/countm
        mean_cbgr[1] = np.sum(imggc)/countm
        mean_cbgr[2] = np.sum(imgrc)/countm
        std_cbgr[0] = np.sqrt(np.sum(np.square(imgbc-mean_cbgr[0]))/countm)
        std_cbgr[1] = np.sqrt(np.sum(np.square(imggc-mean_cbgr[1]))/countm)
        std_cbgr[2] = np.sqrt(np.sum(np.square(imgrc-mean_cbgr[2]))/countm)

        #standard deviation and mean for subgoal image
        #print(goal_img.shape)
        imgrt = (np.reshape(goal_img[0][0],(1,128,128)) + 1)*0.5
        imggt = (np.reshape(goal_img[0][1],(1,128,128)) + 1)*0.5
        imgbt = (np.reshape(goal_img[0][2],(1,128,128)) + 1)*0.5
        mean_tbgr = np.zeros((3,1))
        std_tbgr = np.zeros((3,1))
        mean_tbgr[0] = np.sum(imgbt)/countm
        mean_tbgr[1] = np.sum(imggt)/countm
        mean_tbgr[2] = np.sum(imgrt)/countm
        std_tbgr[0] = np.sqrt(np.sum(np.square(imgbt-mean_tbgr[0]))/countm)
        std_tbgr[1] = np.sqrt(np.sum(np.square(imggt-mean_tbgr[1]))/countm)
        std_tbgr[2] = np.sqrt(np.sum(np.square(imgrt-mean_tbgr[2]))/countm)

        mean_ct[0] = mean_cbgr[0]
        mean_ct[1] = mean_cbgr[1]
        mean_ct[2] = mean_cbgr[2]
        std_ct[0] = std_cbgr[0]
        std_ct[1] = std_cbgr[1]
        std_ct[2] = std_cbgr[2]

        xcg = torch.clamp(torch.from_numpy(cur_img).to(device), -1.0, 1.0)
        #xcg_360 = torch.clamp(torch.from_numpy(cur_img_360).to(device), -1.0, 1.0)
        #xcgf_360 = xcg_360[:, :, :, 0:rsizex]
        #xcgb_360 = xcg_360[:, :, :, rsizex:2*rsizex].flip(3)
        #xcg_360 = torch.cat((xcgf_360, xcgb_360),axis=1)

        imgrtt = (imgrt-mean_tbgr[0])/std_tbgr[0]*std_ct[0]+mean_ct[0]
        imggtt = (imggt-mean_tbgr[1])/std_tbgr[1]*std_ct[1]+mean_ct[1]
        imgbtt = (imgbt-mean_tbgr[2])/std_tbgr[2]*std_ct[2]+mean_ct[2]
        #goalt_img = np.array((np.reshape(np.concatenate((imgrtt, imggtt, imgbtt), axis = 0), (1,3,128,128)) - 0.5)*2.0, dtype=np.float32)
        goalt_img = np.array((np.reshape(np.concatenate((imgrt, imggt, imgbt), axis = 0), (1,3,128,128)) - 0.5)*2.0, dtype=np.float32)
        timage = torch.clamp(torch.from_numpy(goalt_img).to(device), -1.0, 1.0)

        #current image
        xcgx = xcg

        #subgoal image
        xpgb = timage

        rx_size = 0.3#0.3
        robot_size = rx_size*torch.ones(1, 1, 1, 1).to(device)
        r_size = 0.3#0.18
        robot_sizef = r_size*torch.ones(1, 1, 1, 1).to(device)
        #robot_size = r_size*torch.rand(1, 1, 1, 1).to(device)
        px = torch.zeros(1).to(device)
        pz = torch.zeros(1).to(device)
        ry = torch.zeros(1).to(device)

        with torch.no_grad():
            vwres, ptrav, ptravf = polinet(torch.cat((xcgx, xpgb), axis=1), robot_size, robot_sizef, px, pz, ry)
            #vwres = polinet(torch.cat((xcgx, xpgb), axis=1))
            #print(xcgx.size(), vwres.size())
            #pred_future = vunet(xcg_360, vwres)

        for i in range(1):
            dis = torch.sum(vwres[i,0:16:2]*0.333)
            ang = torch.sum(vwres[i,1:16:2]*0.333)
        print("distance", dis)
        print("angle", ang)

        msg_pub = Twist()
        msg_raw = Twist()

        vt = vwres.cpu().numpy()[0,0,0,0]
        wt = vwres.cpu().numpy()[0,1,0,0]
        print(vt, wt, ptravf[0,0])

        #if ptravf[0,0] > 0.5:
        #    vt = 0.0
        
        #vt = 0.0
        #wt = 0.0

        msg_raw.linear.x = vt
        msg_raw.linear.y = 0.0
        msg_raw.linear.z = 0.0
        msg_raw.angular.x = 0.0
        msg_raw.angular.y = 0.0
        msg_raw.angular.z = wt

        maxv = 0.2
        maxw = 0.4

        if np.absolute(vt) < maxv:
            if np.absolute(wt) < maxw:
                msg_pub.linear.x = vt
                msg_pub.linear.y = 0.0
                msg_pub.linear.z = 0.0
                msg_pub.angular.x = 0.0
                msg_pub.angular.y = 0.0
                msg_pub.angular.z = wt
            else:
                rd = vt/wt
                msg_pub.linear.x = maxw * np.sign(vt) * np.absolute(rd)
                msg_pub.linear.y = 0.0
                msg_pub.linear.z = 0.0
                msg_pub.angular.x = 0.0
                msg_pub.angular.y = 0.0
                msg_pub.angular.z = maxw * np.sign(wt)
        else:
            if np.absolute(wt) < 0.001:
                msg_pub.linear.x = maxv * np.sign(vt)
                msg_pub.linear.y = 0.0
                msg_pub.linear.z = 0.0
                msg_pub.angular.x = 0.0
                msg_pub.angular.y = 0.0
                msg_pub.angular.z = 0.0
            else:
                rd = vt/wt
                if np.absolute(rd) > maxv / maxw:
                    msg_pub.linear.x = maxv * np.sign(vt)
                    msg_pub.linear.y = 0.0
                    msg_pub.linear.z = 0.0
                    msg_pub.angular.x = 0.0
                    msg_pub.angular.y = 0.0
                    msg_pub.angular.z = maxv * np.sign(wt) / np.absolute(rd)
                else:
                    msg_pub.linear.x = maxw * np.sign(vt) * np.absolute(rd)
                    msg_pub.linear.y = 0.0
                    msg_pub.linear.z = 0.0
                    msg_pub.angular.x = 0.0
                    msg_pub.angular.y = 0.0
                    msg_pub.angular.z = maxw * np.sign(wt)

        #pred image visualization
        #imgb = np.fmin(255.0, np.fmax(0.0, pred_8*127.5+127.5))
        #imgc = np.reshape(imgb, (3, 128, 256))
        #imgd = imgc.transpose(1, 2, 0)
        #imge = imgd.astype(np.uint8)
        #imgm = bridge.cv2_to_imgmsg(imge, 'bgr8')
        #image_pred.publish(imgm)

        #imgb = np.fmin(255.0, np.fmax(0.0, goal_img_360*127.5+127.5))
        #imgc = np.reshape(imgb, (3, 128, 256))
        #imgd = imgc.transpose(1, 2, 0)
        #imge = imgd.astype(np.uint8)
        #imgm = bridge.cv2_to_imgmsg(imge, 'bgr8')
        #image_ref.publish(imgm)

        imgb = np.fmin(255.0, np.fmax(0.0, xpgb.detach().cpu().numpy()*127.5+127.5))#xcgx xpgb
        imgc = np.reshape(imgb, (3, 128, 128))
        imgd = imgc.transpose(1, 2, 0)
        imge = imgd.astype(np.uint8)
        imgm = bridge.cv2_to_imgmsg(imge, 'bgr8')
        image_ref_input.publish(imgm)

        imgb = np.fmin(255.0, np.fmax(0.0, xcgx.detach().cpu().numpy()*127.5+127.5))#xcgx xpgb
        imgc = np.reshape(imgb, (3, 128, 128))
        imgd = imgc.transpose(1, 2, 0)
        imge = imgd.astype(np.uint8)
        imgm = bridge.cv2_to_imgmsg(imge, 'bgr8')
        image_cur_input.publish(imgm)

        #velocities
        if not isgoal:
                msg_out.publish(msg_pub)
        else:
            if msg_pub.linear.x > 0.05:
                msg_out.publish(msg_pub)

        topic_velcmd = msg_pub

        j = 0
        msg_out_raw.publish(msg_raw)

def callback_fisheye(msg_1):
    global i
    global j
    #global timagev
    #global Nline
    #global count
    #global Lpast
    #global inputgpu
    #global prefv
    #global opt_biasv
    #global swopt
    #global Lmin
    global vwkeep
    #global Lth
    #global mask_brrc
    #global imgbt, imgrt, imggt   
    #global prev_cmd_vel 
    #global goal_img
    #print("kokonihakiteru??", j)
    j = j + 1
    #print(j, goal_img.shape)
    #print
    if j == 1:
        cur_img = preprocess_image_gen(msg_1) #current image
        
        #cur_img_360 = cur_img
        #goal_img_360 = goal_img

        #standard deviation and mean for current image
        imgbc = (np.reshape(cur_img[0][0],(1,128,128)) + 1.0)*0.5
        imggc = (np.reshape(cur_img[0][1],(1,128,128)) + 1.0)*0.5
        imgrc = (np.reshape(cur_img[0][2],(1,128,128)) + 1.0)*0.5
        mean_cbgr = np.zeros((3,1))
        std_cbgr = np.zeros((3,1))
        mean_ct = np.zeros((3,1))
        std_ct = np.zeros((3,1))
        mean_cbgr[0] = np.sum(imgbc)/countm
        mean_cbgr[1] = np.sum(imggc)/countm
        mean_cbgr[2] = np.sum(imgrc)/countm
        std_cbgr[0] = np.sqrt(np.sum(np.square(imgbc-mean_cbgr[0]))/countm)
        std_cbgr[1] = np.sqrt(np.sum(np.square(imggc-mean_cbgr[1]))/countm)
        std_cbgr[2] = np.sqrt(np.sum(np.square(imgrc-mean_cbgr[2]))/countm)

        #standard deviation and mean for subgoal image
        #print(goal_img.shape)
        imgrt = (np.reshape(goal_img[0][0],(1,128,128)) + 1)*0.5
        imggt = (np.reshape(goal_img[0][1],(1,128,128)) + 1)*0.5
        imgbt = (np.reshape(goal_img[0][2],(1,128,128)) + 1)*0.5
        mean_tbgr = np.zeros((3,1))
        std_tbgr = np.zeros((3,1))
        mean_tbgr[0] = np.sum(imgbt)/countm
        mean_tbgr[1] = np.sum(imggt)/countm
        mean_tbgr[2] = np.sum(imgrt)/countm
        std_tbgr[0] = np.sqrt(np.sum(np.square(imgbt-mean_tbgr[0]))/countm)
        std_tbgr[1] = np.sqrt(np.sum(np.square(imggt-mean_tbgr[1]))/countm)
        std_tbgr[2] = np.sqrt(np.sum(np.square(imgrt-mean_tbgr[2]))/countm)

        mean_ct[0] = mean_cbgr[0]
        mean_ct[1] = mean_cbgr[1]
        mean_ct[2] = mean_cbgr[2]
        std_ct[0] = std_cbgr[0]
        std_ct[1] = std_cbgr[1]
        std_ct[2] = std_cbgr[2]

        xcg = torch.clamp(torch.from_numpy(cur_img).to(device), -1.0, 1.0)
        #xcg_360 = torch.clamp(torch.from_numpy(cur_img_360).to(device), -1.0, 1.0)
        #xcgf_360 = xcg_360[:, :, :, 0:rsizex]
        #xcgb_360 = xcg_360[:, :, :, rsizex:2*rsizex].flip(3)
        #xcg_360 = torch.cat((xcgf_360, xcgb_360),axis=1)

        imgrtt = (imgrt-mean_tbgr[0])/std_tbgr[0]*std_ct[0]+mean_ct[0]
        imggtt = (imggt-mean_tbgr[1])/std_tbgr[1]*std_ct[1]+mean_ct[1]
        imgbtt = (imgbt-mean_tbgr[2])/std_tbgr[2]*std_ct[2]+mean_ct[2]
        #goalt_img = np.array((np.reshape(np.concatenate((imgrt, imggt, imgbt), axis = 0), (1,3,128,128)) - 0.5)*2.0, dtype=np.float32)
        goalt_img = np.array((np.reshape(np.concatenate((imgrtt, imggtt, imgbtt), axis = 0), (1,3,128,128)) - 0.5)*2.0, dtype=np.float32)
        timage = torch.clamp(torch.from_numpy(goalt_img).to(device), -1.0, 1.0)

        #current image
        xcgx = 2.0*(mask_recon_polinet[0]*(0.5*xcg+0.5)-0.5)

        #subgoal image
        xpgb = 2.0*(mask_recon_polinet[0]*(0.5*timage+0.5)-0.5)

        r_size = 0.0
        robot_size = r_size*torch.ones(1, 1, 1, 1).to(device)
        #robot_size = r_size*torch.rand(1, 1, 1, 1).to(device)
        px = torch.zeros(1).to(device)
        pz = torch.zeros(1).to(device)
        ry = torch.zeros(1).to(device)

        with torch.no_grad():
            vwres, ptrav, ptravf = polinet(torch.cat((xcgx, xpgb), axis=1), robot_size, robot_size, px, pz, ry)
            #vwres = polinet(torch.cat((xcgx, xpgb), axis=1))
            #print(xcgx.size(), vwres.size())
            #pred_future = vunet(xcg_360, vwres)

        for i in range(1):
            dis = torch.sum(vwres[i,0:16:2]*0.333)
            ang = torch.sum(vwres[i,1:16:2]*0.333)
        print("distance", dis)
        print("angle", ang)

        msg_pub = Twist()
        msg_raw = Twist()

        vt = vwres.cpu().numpy()[0,0,0,0]
        wt = vwres.cpu().numpy()[0,1,0,0]
        print(vt, wt)
        #vt = 0.0
        #wt = 0.0

        msg_raw.linear.x = vt
        msg_raw.linear.y = 0.0
        msg_raw.linear.z = 0.0
        msg_raw.angular.x = 0.0
        msg_raw.angular.y = 0.0
        msg_raw.angular.z = wt

        maxv = 0.2
        maxw = 0.4

        if np.absolute(vt) < maxv:
            if np.absolute(wt) < maxw:
                msg_pub.linear.x = vt
                msg_pub.linear.y = 0.0
                msg_pub.linear.z = 0.0
                msg_pub.angular.x = 0.0
                msg_pub.angular.y = 0.0
                msg_pub.angular.z = wt
            else:
                rd = vt/wt
                msg_pub.linear.x = maxw * np.sign(vt) * np.absolute(rd)
                msg_pub.linear.y = 0.0
                msg_pub.linear.z = 0.0
                msg_pub.angular.x = 0.0
                msg_pub.angular.y = 0.0
                msg_pub.angular.z = maxw * np.sign(wt)
        else:
            if np.absolute(wt) < 0.001:
                msg_pub.linear.x = maxv * np.sign(vt)
                msg_pub.linear.y = 0.0
                msg_pub.linear.z = 0.0
                msg_pub.angular.x = 0.0
                msg_pub.angular.y = 0.0
                msg_pub.angular.z = 0.0
            else:
                rd = vt/wt
                if np.absolute(rd) > maxv / maxw:
                    msg_pub.linear.x = maxv * np.sign(vt)
                    msg_pub.linear.y = 0.0
                    msg_pub.linear.z = 0.0
                    msg_pub.angular.x = 0.0
                    msg_pub.angular.y = 0.0
                    msg_pub.angular.z = maxv * np.sign(wt) / np.absolute(rd)
                else:
                    msg_pub.linear.x = maxw * np.sign(vt) * np.absolute(rd)
                    msg_pub.linear.y = 0.0
                    msg_pub.linear.z = 0.0
                    msg_pub.angular.x = 0.0
                    msg_pub.angular.y = 0.0
                    msg_pub.angular.z = maxw * np.sign(wt)

        #pred image visualization
        #imgb = np.fmin(255.0, np.fmax(0.0, pred_8*127.5+127.5))
        #imgc = np.reshape(imgb, (3, 128, 256))
        #imgd = imgc.transpose(1, 2, 0)
        #imge = imgd.astype(np.uint8)
        #imgm = bridge.cv2_to_imgmsg(imge, 'bgr8')
        #image_pred.publish(imgm)

        #imgb = np.fmin(255.0, np.fmax(0.0, goal_img_360*127.5+127.5))
        #imgc = np.reshape(imgb, (3, 128, 256))
        #imgd = imgc.transpose(1, 2, 0)
        #imge = imgd.astype(np.uint8)
        #imgm = bridge.cv2_to_imgmsg(imge, 'bgr8')
        #image_ref.publish(imgm)

        imgb = np.fmin(255.0, np.fmax(0.0, xpgb.detach().cpu().numpy()*127.5+127.5))#xcgx xpgb
        imgc = np.reshape(imgb, (3, 128, 128))
        imgd = imgc.transpose(1, 2, 0)
        imge = imgd.astype(np.uint8)
        imgm = bridge.cv2_to_imgmsg(imge, 'bgr8')
        image_cur_input.publish(imgm)

        #velocities
        if not isgoal:
                msg_out.publish(msg_pub)
        else:
            if msg_pub.linear.x > 0.05:
                msg_out.publish(msg_pub)

        topic_velcmd = msg_pub

        j = 0
        msg_out_raw.publish(msg_raw)

def callback_state_lattice():
    global batch_data
    
    if batch_data != {}:    
        
        #goal pose mat            
        #Ttrans = self.prev_mat + (bl+1.0)*(obs["trans_mat"] - self.prev_mat)/len(self.buffer_loc)
        Ttrans = batch_data["trans_mat"]
        T265_cur = batch_data["t265_odom_cur"]    
        Tgoal_n = batch_data["goal_pose"]     
                
        Traj_n = np.dot(Ttrans, T265_cur)
          
        axes_input = 'szyz'
        a1_traj, a2_traj, a3_traj = mat2euler(Traj_n[0:3, 0:3], axes=axes_input)

        Traj = np.zeros((4, 4))                                           
        Traj[0, 0] = np.cos(-a3_traj)
        Traj[0, 2] = np.sin(-a3_traj)
        Traj[1, 1] = 1.0
        Traj[2, 0] = -np.sin(-a3_traj)
        Traj[2, 2] = np.cos(-a3_traj)
        Traj[0, 3] = Traj_n[1, 3]
        Traj[2, 3] = -Traj_n[0, 3]
        Traj[3, 3] = 1.0
        
        a1_goal, a2_goal, a3_goal = mat2euler(Tgoal_n[0:3, 0:3], axes=axes_input)    
                    
        Tgoal = np.zeros((4, 4))                                           
        Tgoal[0, 0] = np.cos(-a3_goal)
        Tgoal[0, 2] = np.sin(-a3_goal)
        Tgoal[1, 1] = 1.0
        Tgoal[2, 0] = -np.sin(-a3_goal)
        Tgoal[2, 2] = np.cos(-a3_goal)
        Tgoal[0, 3] = Tgoal_n[1, 3]
        Tgoal[2, 3] = -Tgoal_n[0, 3]
        Tgoal[3, 3] = 1.0
        Tgoal_local = np.dot(np.linalg.inv(Traj), Tgoal)    
        """
        a3_goal = 0.5*np.pi/2.0
        Tgoal_local = np.zeros((4, 4))                                           
        Tgoal_local[0, 0] = np.cos(-a3_goal)
        Tgoal_local[0, 2] = np.sin(-a3_goal)
        Tgoal_local[1, 1] = 1.0
        Tgoal_local[2, 0] = -np.sin(-a3_goal)
        Tgoal_local[2, 2] = np.cos(-a3_goal)
        Tgoal_local[0, 3] = -1.0
        Tgoal_local[2, 3] = 2.0
        Tgoal_local[3, 3] = 1.0
        """
        
        Tref = torch.from_numpy(Tgoal_local.copy())              
        Tref[0, 3] = 2.0*torch.from_numpy(Tgoal_local.copy())[0, 3]
        Tref[2, 3] = 2.0*torch.from_numpy(Tgoal_local.copy())[2, 3]
        
        bsl, lsl, _, _ = Tprem.size()
        #eval_lattice = lattice_mse(Tref.unsqueeze(0).repeat(bsl,1,1), Tprem)
        #eval_lattice = torch.mean((Tref[0:3,0:3].unsqueeze(0).repeat(bsl,1,1)-Tprem[:,0:3,0:3])**2, dim=(1,2))
    
        #print(Tref.unsqueeze(0).repeat(bsl,1,1).size(), Tprem[:,lsl-1].size())
        Tprem_split = torch.split(Tprem, 1, dim=1)
        Tprem_split_cat = torch.cat(Tprem_split, dim=0).squeeze(1)
        
        #print("split", Tprem_split[0].size(), Tprem_split_cat.size())
        eval_lattice = torch.mean((Tref.unsqueeze(0).repeat(bsl*lsl,1,1)-Tprem_split_cat)**2, dim=(1,2))

    
        offset = 10
        offset_m = 5
        pc_front_x = torch.flatten(torch.from_numpy(batch_data["pc_next"])[0,64-offset_m:64+offset_m,offset:128-offset], start_dim=0, end_dim=1)
        pc_front_z = torch.flatten(torch.from_numpy(batch_data["pc_next"])[2,64-offset_m:64+offset_m,offset:128-offset], start_dim=0, end_dim=1)
        pc_front_x_batch = pc_front_x.unsqueeze(0).repeat(bsl*lsl, 1)
        pc_front_z_batch = pc_front_z.unsqueeze(0).repeat(bsl*lsl, 1)
                
        x_traj = 0.5*Tprem_split_cat[:, 0, 3].unsqueeze(1)
        z_traj = 0.5*Tprem_split_cat[:, 2, 3].unsqueeze(1)       
        
        #print(px_torch_cat.size(), pz_torch_cat.size(), x_traj.size(), z_traj.size(), pc_front_x_batch.size(), pc_front_z_batch.size(), offset_m*2)
        #print("xz value", x_traj[91], z_traj[91])
        radius = 0.5
        collision_judge = torch.sum(((pc_front_x_batch - x_traj)**2 + (pc_front_z_batch - z_traj)**2 < radius**2).float(), dim=1) > 5.0 # > 5.0
        #print("split", torch.split(collision_judge, bsl, dim=0))
        #print("split", sum(torch.split(collision_judge, bsl, dim=0)))
        collision_judge = (sum(torch.split(collision_judge, bsl, dim=0))> 0.5).float().repeat(lsl) # 
        
        #print(sum(collision_judge).size(), sum(collision_judge))
        #print(sum(((pc_front_x_batch - x_traj)**2 + (pc_front_z_batch - z_traj)**2 < radius**2)[1].float()))
        #print(torch.sum(((pc_front_x_batch - x_traj)**2 + (pc_front_z_batch - z_traj)**2 < radius**2).float(), dim=1))

        #ped estimation
        robot_future_xy = torch.zeros(1,16).float().cuda()
        
        norm = 10.0
        #ped_past_xy = torch.clamp(torch.from_numpy(batch_data["ped_traj"]).to(device), min=-norm, max=norm)   #pedestrian's past traj.
        #robot_past_xy = torch.from_numpy(batch_data["robot_traj"]).to(device)                                 #robot's past traj.

        ped_n_xy, robot_n_xy = ped_transform(batch_data["rob_traj"], batch_data["ped_traj"])
        
        #test sample
        #robot_n_xy = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7])
        #ped_n_xy = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 2.0, 2.333, 2.666, 3.000, 3.333, 3.666, 4.0, 4.333])
        #ped_n_xy = np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 2.0, 2.333, 2.666, 3.000, 3.333, 3.666, 4.0, 4.333])
                
        ped_n_xy_norm = torch.clamp(torch.from_numpy(ped_n_xy).cuda(), min=-10.0, max=10.0)/norm  
        robot_n_xy_norm = torch.clamp(torch.from_numpy(robot_n_xy).cuda(), min=-10.0, max=10.0)/norm  
        
        #print(ped_n_xy_norm.size(), robot_n_xy_norm.size(), robot_future_xy.size())
        with torch.no_grad():
            delta_est_ped_future = pednet(ped_n_xy_norm.unsqueeze(0).float(), robot_n_xy_norm.unsqueeze(0).float(), robot_future_xy/norm) 

        ped_future_x = torch.cumsum(delta_est_ped_future[:,0:8]/norm, dim=1) + torch.from_numpy(ped_n_xy).cuda().float().unsqueeze(0)[:,0:1].repeat(1,8)/norm
        ped_future_y = torch.cumsum(delta_est_ped_future[:,8:16]/norm, dim=1) + torch.from_numpy(ped_n_xy).cuda().float().unsqueeze(0)[:,8:9].repeat(1,8)/norm           
        ped_future_xy = torch.clamp(torch.cat((ped_future_x, ped_future_y), axis=1), min=-norm, max=norm)*norm                          #estimated pedestrian's traj.
        #ax4.plot(ped_future_xy[0,0:8].cpu().numpy(), ped_future_xy[0,8:16].cpu().numpy(), color='orange', marker="o", label='ped. future')   
        
        px_ped_f = norm*ped_future_x.repeat(bsl, 1)
        pz_ped_f = norm*ped_future_y.repeat(bsl, 1)
        
        eval_ped = (torch.sum((px_ped_f - px_torch_cat.cuda())**2 + (pz_ped_f - pz_torch_cat.cuda())**2 < (0.2 + 0.5)**2, dim=1) > 0.5).repeat(lsl).float().cpu()
        
        #print(collision_judge)
        #print(eval_ped)
        ped_cur =  torch.sum(torch.from_numpy(ped_n_xy).cuda().float().unsqueeze(0)[:,0:1] + torch.from_numpy(ped_n_xy).cuda().float().unsqueeze(0)[:,8:9])
        #if torch.sum(delta_est_ped_future) < 0.0000001:
        if ped_cur < 0.0000001:
           no_ped = 0.0
        else:
           no_ped = 1.0
        
        #no_ped = 0.0  
        metric = eval_lattice + 1000.0*collision_judge + 1000.0 * eval_ped * no_ped
        
        print("no pedestrians flag", no_ped)
        #print(metric)        
        id_min = torch.argmin(metric)
        id_select = id_min % bsl
        #print("eval_lattice", eval_lattice.size(), id_min, id_select, eval_lattice)
        #print("linear vel.", vel_prem[id_select][0], "angular vel.", vel_prem[id_select][1])
    
        if metric[id_min] < 999:
            linear_vel = vel_prem[id_select][0].cpu().numpy()
            angular_vel = vel_prem[id_select][1].cpu().numpy()
        else:
            linear_vel = 0.0
            angular_vel = 0.0         

        """
        vis_x = torch.from_numpy(batch_data["pc_next"])[0,64-offset_m:64+offset_m,offset:128-offset]
        vis_z = torch.from_numpy(batch_data["pc_next"])[2,64-offset_m:64+offset_m,offset:128-offset]
        for i in range(bvp):
            plt.plot(px_torch_cat[i].cpu().numpy(), pz_torch_cat[i].cpu().numpy(), color='blue', marker = "x")
        
          
        plt.plot(px_torch_cat[id_select].cpu().numpy(), pz_torch_cat[id_select].cpu().numpy(), linewidth=3.0, color='red', marker = "o", label="selected traj")     
        plt.plot(robot_n_xy[0:8], robot_n_xy[8:16], color='magenta', marker = "x", label="robot past traj")  
        #plt.plot(ped_n_xy[0:8], ped_n_xy[8:16], linewidth=3.0, color='orange', marker = "s", label="ped. past traj")         
        #plt.plot(norm*ped_future_x.squeeze(0).cpu().numpy(), norm*ped_future_y.squeeze(0).cpu().numpy(), linewidth=3.0, color='green', marker = "s", label="estimated ped. traj")   
        plt.plot(Tgoal_local[0, 3], Tgoal_local[2, 3], color='red', marker='*', label="goal position")
        
        for j in range(offset_m*2):            
            plt.plot(vis_x[j].cpu().numpy(), vis_z[j].cpu().numpy(), color="black", marker = ".")
            
        plt.xlim(-1.8, 1.2)
        plt.ylim(-0.9, 4.9)
        plt.legend(loc="upper left")
        plt.show()  
        """
        
        #print(collision_judge.size(), pc_front_x_batch.size(), pc_front_z_batch.size(), x_traj.size(), z_traj.size())        
        #print("pc_front", pc_front_x.size())
        #print("judge_before_sum", (sum(torch.split(collision_judge, bsl, dim=0))).float()/lsl)
    else:
        linear_vel = 0.0
        angular_vel = 0.0   
        print("no data!")
    
    return linear_vel, angular_vel
    #point clouds
    #batch_data["pc_cur"] 

def visualize_control(image, px_ref_list, pz_ref_list):

    print(len(px_ref_list), px_ref_list[0].size(), len(pz_ref_list), pz_ref_list[0].size())
    
    x = []
    z = []
    for i in range(len(px_ref_list)):
        x.append(px_ref_list[i].item())
        z.append(pz_ref_list[i].item())
        
    fig = plt.figure()
    gs = fig.add_gridspec(1,2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    
    ax1.imshow(image)
    ax2.plot(x, z, marker="o")
    ax2.axis(ymin=-0.5, ymax=6.0)
    ax2.axis(xmin=-3.0,xmax=3.0)

    plt.show()       

    
def callback_360(msg_1):
    global icount
    global bag_file
    global i
    global j
    global timagev
    global Nline
    global count
    global Lpast
    global inputgpu
    global prefv
    global opt_biasv
    global swopt
    global Lmin
    global vwkeep
    global Lth
    global mask_brrc
    global imgbt, imgrt, imggt   
    global prev_cmd_vel 
    global em_stop, vjoy, wjoy
    global cur_img_raw
    global init_hist, store_hist, image_hist
    global xhist1_x, xhist2_x, xhist3_x, xhist4_x, xhist5_x, xhist6_x, xhist7_x
    #global xhist1_y, xhist2_y, xhist3_y, xhist4_y, xhist5_y, xhist6_y
    global xgoal_x
    #global xgoal_y
    #global current_obs_x
    #global current_obs_y
    global vel_x, vel_previous
    #global vel_y
    global cur_trans_mat_np, cur_t265_mat_np
    global left_bumper, right_bumper, count_comeback
    global markerArray
    global topic_velcmd
    global flag_slow
    global counter_trans, total_count
    global v_prev, w_prev
    
    global rob9_x, rob8_x, rob7_x, rob6_x, rob5_x, rob4_x, rob3_x, rob2_x, rob1_x
    global rob9_y, rob8_y, rob7_y, rob6_y, rob5_y, rob4_y, rob3_y, rob2_y, rob1_y
    global rob9_yaw, rob8_yaw, rob7_yaw, rob6_yaw, rob5_yaw, rob4_yaw, rob3_yaw, rob2_yaw, rob1_yaw
    global ped9_x, ped8_x, ped7_x, ped6_x, ped5_x, ped4_x, ped3_x, ped2_x, ped1_x
    global ped9_z, ped8_z, ped7_z, ped6_z, ped5_z, ped4_z, ped3_z, ped2_z, ped1_z       
    global pc_next, pc_small
    global vel_vec
    
    global batch_data
    global flag_once
    global feat_text
    global obj_inst
    global id_goal
    #print("test", j)
    #global device
    j = j + 1
    #print j
    #print("test", j)
    if j == 1:
        #linear_vel, angular_vel = callback_state_lattice()
        newsize = (96, 96)    
        context_size = 5
        im = msg_to_pil(msg_1)
        #print(im.size)
        im_crop_o = im.crop((50, 100, 1230, 860))
        #im_crop_o = im.crop((25, 50, 615, 430))        
        im_crop = im_crop_o.resize(newsize, PILImage.Resampling.LANCZOS).convert('RGB')     
        #print(np.array(im_crop).shape)


        if init_hist == 0:
            print("init only")
            for ih in range(context_size + 1):
                image_hist.append(im_crop)
            init_hist = 1
        
        if context_size is not None:
            if len(image_hist) < context_size + 1:
                image_hist.append(im_crop)
            else:
                image_hist.pop(0)
                image_hist.append(im_crop)
                        
        #im = msg_to_pil(msg_1)
        #im_crop = im.resize((560, 560), PILImage.Resampling.LANCZOS)
                
        #if init_hist == 0:
        #    print("init only")
        #    for ih in range(10):
        #        image_hist.append(im_crop)
        #    init_hist = 1
        #        
        #im_obs = image_hist[5:10] + [im_crop]
        #obs_images, obs_current_large = transform_images_noriaki3(im_obs, model_params["image_size"], center_crop=False)   

        obs_images = transform_images_exaug(image_hist)
        obs_images = torch.split(obs_images, 3, dim=1)
        obs_images = torch.cat(obs_images, dim=1) 
        batch_obs_images = obs_images.to(device)
          
        #for poise t265
        t265_vec = [0, 0, 0, 0, 0, 0, 1]
        t265_vec[0] = topic_t265.pose.pose.position.x
        t265_vec[1] = topic_t265.pose.pose.position.y
        t265_vec[2] = topic_t265.pose.pose.position.z
        t265_vec[3] = topic_t265.pose.pose.orientation.x
        t265_vec[4] = topic_t265.pose.pose.orientation.y
        t265_vec[5] = topic_t265.pose.pose.orientation.z
        t265_vec[6] = topic_t265.pose.pose.orientation.w
		
        t265_mat = quat2rotmat(xyzw2wxyz(t265_vec[3:7]))
        t265_mat[0,3] = t265_vec[0]
        t265_mat[1,3] = t265_vec[1]
        t265_mat[2,3] = t265_vec[2]	        
        t265_mat_np = np.array(t265_mat, dtype=np.float32)
        r = R.from_matrix(t265_mat_np[0:3, 0:3])
        euler_angles = r.as_euler('zyx')
        
        #euler_angles_x = transforms3d.euler.quat2euler(t265_vec[3:7])
        euler_angles_x = transforms3d.euler.quat2euler(xyzw2wxyz(t265_vec[3:7]))        
        #x_position = t265_vec[1]
        #y_position = -t265_vec[0]
        x_position = -t265_vec[0]
        y_position = -t265_vec[1]        
        
        #x_position = 1.0
        #y_position = -1.0
        #yaw_angle = euler_angles[1] + 0.5*3.141592    
        yaw_angle = euler_angles[2]    
        #yaw_angle = 0.5*3.141592   
        #print("robot pose", x_position, y_position, yaw_angle)
        #print("euler_angles", euler_angles)
        #print("quat_angles", t265_vec[3:7])     
        
        # Initial and final quaternion poses
        initial_pose = [0.0, -0.69, 0.0, 0.717]  # initial quaternion
        final_pose = t265_vec[3:7]  # final quaternion

        # Compute the inverse of the initial quaternion
        initial_inverse = quat_inverse(initial_pose)

        # Compute the relative quaternion by multiplying final quaternion with the inverse of initial quaternion
        relative_quaternion = quat_multiply(final_pose, initial_inverse)

        # Calculate the yaw (angle around the z-axis) from the relative quaternion
        yaw_angle = quat_to_yaw(relative_quaternion)

        # Extract the yaw (rotation around the z-axis) from the relative quaternion
        #yaw_angle = math.atan2(2.0 * (relative_rotation[3] * relative_rotation[2] + relative_rotation[0] * relative_rotation[1]), 1.0 - 2.0 * (relative_rotation[1]**2 + relative_rotation[2]**2))        
        print("robot pose", x_position, y_position, yaw_angle)
        
        with torch.no_grad():                        
            #print(obj_inst, feat_text)
            B = batch_obs_images.shape[0]

            metric_waypoint_spacing = 0.25
            loc_pos = True
            if loc_pos:
                relative_pose = calc_relative_pose([x_position, y_position, yaw_angle], xy_subgoal[id_goal] + [yaw_subgoal[id_goal]])

                thres_dist = 30.0
                thres_update = 2.0 
                if np.sqrt(relative_pose[0]**2 + relative_pose[1]**2) > thres_dist:
                    relative_x = relative_pose[0]/np.sqrt(relative_pose[0]**2 + relative_pose[1]**2)*thres_dist
                    relative_y = relative_pose[1]/np.sqrt(relative_pose[0]**2 + relative_pose[1]**2)*thres_dist   
                else:
                    relative_x = relative_pose[0]
                    relative_y = relative_pose[1] 
                    
                relative_ang = relative_pose[2]               
                goal_pose = np.array([relative_x/metric_waypoint_spacing, relative_y/metric_waypoint_spacing, np.cos(relative_ang), np.sin(relative_ang)])
                
                #if True and id_goal != len(yaw_subgoal) - 1:                  
                if np.sqrt(relative_x**2 + relative_y**2) < thres_update and id_goal != len(yaw_subgoal) - 1:              
                    id_goal += 1                     
                    
            else:
                goal_pose = np.array([100.0, 0.0, 1.0, 0.0])            
            
            goal_pose_torch = torch.from_numpy(goal_pose).unsqueeze(0).float().to(device)
            
            print("robot pose", x_position, y_position, yaw_angle)
            print("relative pose", goal_pose[0]*metric_waypoint_spacing, goal_pose[1]*metric_waypoint_spacing, goal_pose[2], goal_pose[3], id_goal)

            
            print(batch_obs_images.size(), goal_pose_torch.size())
            with torch.no_grad():  
                waypoints = model(batch_obs_images, goal_pose_torch)         
            waypoints = to_numpy(waypoints)
            
        if waypoints is not None:
            #for ig in range(end+1-start):
            if True:  
                chosen_waypoint = waypoints[0][2].copy()

                if True: #if we apply normalization in training
                    MAX_v = 0.3
                    RATE = 3.0
                    chosen_waypoint[:2] *= (MAX_v / RATE)
                
                dx, dy, hx, hy = chosen_waypoint

                EPS = 1e-8 #default value of NoMaD inference
                DT = 1/4 #default value of NoMaD inference
                
                if np.abs(dx) < EPS and np.abs(dy) < EPS:
                    linear_vel_value = 0
                    angular_vel_value = clip_angle(np.arctan2(hy, hx))/DT
                elif np.abs(dx) < EPS:
                    linear_vel_value =  0
                    angular_vel_value = np.sign(dy) * np.pi/(2*DT)
                else:
                    linear_vel_value = dx / DT
                    angular_vel_value = np.arctan(dy/dx) / DT
                linear_vel_value = np.clip(linear_vel_value, 0, 0.5)
                angular_vel_value = np.clip(angular_vel_value, -1.0, 1.0)                                    

        msg_pub = Twist()
        msg_raw = Twist()
        
        print("linear vel", linear_vel_value, "angular_vel", angular_vel_value)
        vt = linear_vel_value
        wt = angular_vel_value

        msg_raw.linear.x = vt
        msg_raw.linear.y = 0.0
        msg_raw.linear.z = 0.0
        msg_raw.angular.x = 0.0
        msg_raw.angular.y = 0.0
        msg_raw.angular.z = wt

        if flag_slow == 0:
            maxv = 0.2
            maxw = 0.2        
        else:
            maxv = 0.1
            maxw = 0.25

        if np.absolute(vt) <= maxv:
            if np.absolute(wt) <= maxw:
                msg_pub.linear.x = vt
                msg_pub.linear.y = 0.0
                msg_pub.linear.z = 0.0
                msg_pub.angular.x = 0.0
                msg_pub.angular.y = 0.0
                msg_pub.angular.z = wt
            else:
                rd = vt/wt
                msg_pub.linear.x = maxw * np.sign(vt) * np.absolute(rd)
                msg_pub.linear.y = 0.0
                msg_pub.linear.z = 0.0
                msg_pub.angular.x = 0.0
                msg_pub.angular.y = 0.0
                msg_pub.angular.z = maxw * np.sign(wt)
        else:
            if np.absolute(wt) <= 0.001:
                msg_pub.linear.x = maxv * np.sign(vt)
                msg_pub.linear.y = 0.0
                msg_pub.linear.z = 0.0
                msg_pub.angular.x = 0.0
                msg_pub.angular.y = 0.0
                msg_pub.angular.z = 0.0
            else:
                rd = vt/wt
                if np.absolute(rd) >= maxv / maxw:
                    msg_pub.linear.x = maxv * np.sign(vt)
                    msg_pub.linear.y = 0.0
                    msg_pub.linear.z = 0.0
                    msg_pub.angular.x = 0.0
                    msg_pub.angular.y = 0.0
                    msg_pub.angular.z = maxv * np.sign(wt) / np.absolute(rd)
                else:
                    msg_pub.linear.x = maxw * np.sign(vt) * np.absolute(rd)
                    msg_pub.linear.y = 0.0
                    msg_pub.linear.z = 0.0
                    msg_pub.angular.x = 0.0
                    msg_pub.angular.y = 0.0
                    msg_pub.angular.z = maxw * np.sign(wt)

        #print(vjoy, wjoy)
        if np.abs(vjoy) > 0.00001 or np.abs(wjoy) > 0.00001:
            msg_pub.linear.x = vjoy
            msg_pub.angular.z = wjoy
        """
        elif int(em_stop) == 0:
            #print("kiteru??")
            msg_pub.linear.x = 0.0
            msg_pub.angular.z = 0.0
        elif int(em_stop2) == 0:
            msg_pub.linear.x = 0.0
            msg_pub.angular.z = 0.0
        """
        #velocities
        if not isgoal:
                msg_out.publish(msg_pub)
        else:
            if msg_pub.linear.x > 0.05:
                msg_out.publish(msg_pub)
        #msg_out.publish(msg_pub)

        j = 0
        
        #print(msg_pub.linear.x, msg_pub.angular.z, vt_org, wt_org)
        #flag_vw = 0
        #if msg_pub.linear.x == vt_org and msg_pub.angular.z == wt_org:
        #    flag_vw = 1
        #print("flag_vw", flag_vw)
        print("infe linear vel", msg_pub.linear.x, "infe angular_vel", msg_pub.angular.z)        
        msg_out_raw.publish(msg_raw)
        topic_velcmd = msg_pub


        #image_histx = [im_crop] + image_hist[0:9]
        #image_hist = image_histx
        #next_obs = torch.cat((xcgx, xhist1_x, xhist2_x, xhist3_x, xhist4_x, xhist5_x, xhist6_x, xgoal_x), axis=1).cpu().numpy()
        ##cur_obs = current_obs_x.cpu().numpy() 
        
        #obs_img = torch.cat((xcgx, xhist1_x, xhist2_x, xhist3_x, xhist4_x, xhist5_x, xhist6_x, xhist7_x), axis=1).cpu().numpy()
        #goal_obs = xgoal_x.cpu().numpy() 
        #actions = vel_x
        #actions_previous = vel_previous

        imgd = np.array(im_crop_o.convert('RGB'))
        imge = imgd.astype(np.uint8)
        imgm = bridge.cv2_to_imgmsg(imge, 'rgb8')
        crop_image.publish(imgm)        
        """
        if store_hist == 0:        
            next_obs = torch.cat((xcgx, xhist1_x, xhist2_x, xhist3_x, xhist4_x, xhist5_x, xhist6_x, xgoal_x), axis=1).cpu().numpy()
            cur_obs = current_obs_x.cpu().numpy() 
            actions = vel_x
        elif store_hist == 1:
            next_obs = torch.cat((xcgx, xhist1_y, xhist2_y, xhist3_y, xhist4_y, xhist5_y, xhist6_y, xgoal_y), axis=1).cpu().numpy() 
            cur_obs = current_obs_y.cpu().numpy()         
            actions = vel_y
        """
        """
        #for trans mat
        trans_vec = [0, 0, 0, 0, 0, 0, 1]
        trans_vec[0] = topic_trans_mat.position.x
        trans_vec[1] = topic_trans_mat.position.y
        trans_vec[2] = topic_trans_mat.position.z
        trans_vec[3] = topic_trans_mat.orientation.x
        trans_vec[4] = topic_trans_mat.orientation.y
        trans_vec[5] = topic_trans_mat.orientation.z
        trans_vec[6] = topic_trans_mat.orientation.w
    	
        trans_mat = quat2rotmat(xyzw2wxyz(trans_vec[3:7]))
        trans_mat[0,3] = trans_vec[0]
        trans_mat[1,3] = trans_vec[1]
        trans_mat[2,3] = trans_vec[2]	            
        trans_mat_np = np.array(trans_mat, dtype=np.float32)
        
        #for poise t265
        t265_vec = [0, 0, 0, 0, 0, 0, 1]
        t265_vec[0] = topic_t265.pose.pose.position.x
        t265_vec[1] = topic_t265.pose.pose.position.y
        t265_vec[2] = topic_t265.pose.pose.position.z
        t265_vec[3] = topic_t265.pose.pose.orientation.x
        t265_vec[4] = topic_t265.pose.pose.orientation.y
        t265_vec[5] = topic_t265.pose.pose.orientation.z
        t265_vec[6] = topic_t265.pose.pose.orientation.w
		
        t265_mat = quat2rotmat(xyzw2wxyz(t265_vec[3:7]))
        t265_mat[0,3] = t265_vec[0]
        t265_mat[1,3] = t265_vec[1]
        t265_mat[2,3] = t265_vec[2]	        
        t265_mat_np = np.array(t265_mat, dtype=np.float32)

        #for trans mat
        gpose_vec = [0, 0, 0, 0, 0, 0, 1]
        gpose_vec[0] = topic_gpose_mat.position.x
        gpose_vec[1] = topic_gpose_mat.position.y
        gpose_vec[2] = topic_gpose_mat.position.z
        gpose_vec[3] = topic_gpose_mat.orientation.x
        gpose_vec[4] = topic_gpose_mat.orientation.y
        gpose_vec[5] = topic_gpose_mat.orientation.z
        gpose_vec[6] = topic_gpose_mat.orientation.w
    	
        gpose_mat = quat2rotmat(xyzw2wxyz(gpose_vec[3:7]))
        gpose_mat[0,3] = gpose_vec[0]
        gpose_mat[1,3] = gpose_vec[1]
        gpose_mat[2,3] = gpose_vec[2]	            
        gpose_mat_np = np.array(gpose_mat, dtype=np.float32)
        """
        """        
        if store_hist == 0:
            xhist6_x = xhist5_x
            xhist5_x = xhist4_x
            xhist4_x = xhist3_x
            xhist3_x = xhist2_x
            xhist2_x = xhist1_x
            xhist1_x = xcgx
            xgoal_x = xpgx
            current_obs_x = image_cat
            vel_x = np.array([msg_pub.linear.x, msg_pub.angular.z], dtype=np.float32)
            store_hist = 1
        elif store_hist == 1:
            xhist6_y = xhist5_y
            xhist5_y = xhist4_y
            xhist4_y = xhist3_y
            xhist3_y = xhist2_y
            xhist2_y = xhist1_y
            xhist1_y = xcgx
            xgoal_y = xpgx
            current_obs_y = image_cat
            vel_y = np.array([msg_pub.linear.x, msg_pub.angular.z], dtype=np.float32)                        
            store_hist = 0
        """
        """
        if sum(sum(cur_trans_mat_np)) == sum(sum(trans_mat_np)):# and counter_trans < 20:
            counter_trans += 1
            #print("A")
        else:
            total_count = counter_trans
            counter_trans = 1
            #print("B")

        #for robot traj in past
        rob9_x = rob8_x
        rob8_x = rob7_x
        rob7_x = rob6_x        
        rob6_x = rob5_x
        rob5_x = rob4_x
        rob4_x = rob3_x
        rob3_x = rob2_x
        rob2_x = rob1_x
        rob1_x = x_pos        

        rob9_y = rob8_y
        rob8_y = rob7_y
        rob7_y = rob6_y
        rob6_y = rob5_y
        rob5_y = rob4_y
        rob4_y = rob3_y
        rob3_y = rob2_y
        rob2_y = rob1_y
        rob1_y = y_pos    

        rob9_yaw = rob8_yaw
        rob8_yaw = rob7_yaw
        rob7_yaw = rob6_yaw
        rob6_yaw = rob5_yaw
        rob5_yaw = rob4_yaw
        rob4_yaw = rob3_yaw
        rob3_yaw = rob2_yaw
        rob2_yaw = rob1_yaw
        rob1_yaw = yaw 
        robot_traj = np.array([[rob1_x, rob2_x, rob3_x, rob4_x, rob5_x, rob6_x, rob7_x, rob8_x, rob9_x], [rob1_y, rob2_y, rob3_y, rob4_y, rob5_y, rob6_y, rob7_y, rob8_y, rob9_y], [rob1_yaw, rob2_yaw, rob3_yaw, rob4_yaw, rob5_yaw, rob6_yaw, rob7_yaw, rob8_yaw, rob9_yaw]], dtype=np.float32)        
        
        #for ped. traj in past
        ped9_x = ped8_x
        ped8_x = ped7_x
        ped7_x = ped6_x        
        ped6_x = ped5_x
        ped5_x = ped4_x
        ped4_x = ped3_x
        ped3_x = ped2_x
        ped2_x = ped1_x
        
        if xpos_close != []:
            ped1_x = xpos_close[0]        
        else:
            ped1_x = 0.0

        ped9_z = ped8_z
        ped8_z = ped7_z
        ped7_z = ped6_z
        ped6_z = ped5_z
        ped5_z = ped4_z
        ped4_z = ped3_z
        ped3_z = ped2_z
        ped2_z = ped1_z     

        if xpos_close != []:
            ped1_z = zpos_close[0]        
        else:
            ped1_z = 0.0
        ped_traj = np.array([[ped1_x, ped2_x, ped3_x, ped4_x, ped5_x, ped6_x, ped7_x, ped8_x, ped9_x], [ped1_z, ped2_z, ped3_z, ped4_z, ped5_z, ped6_z, ped7_z, ped8_z, ped9_z]], dtype=np.float32)            
                    
        batch_data = {
            #"observations": cur_obs,
            "observations": (127.5*obs_img + 127.5).astype(np.uint8),            
            "actions": actions,
            "actions_previous": actions_previous,
            "vel_vec": vel_vec,
            "flag_vw": np.array(flag_vw, dtype=np.float32),
            "flag_col": np.array(flag_col, dtype=np.float32),            
            #"next_observations": next_obs,
            "goal_observations": (127.5*goal_obs + 127.5).astype(np.uint8),            
            "trans_mat": trans_mat_np,     
            "t265_odom": t265_mat_np,
            "trans_mat_cur": cur_trans_mat_np,     
            "t265_odom_cur": cur_t265_mat_np,            
            "goal_pose": gpose_mat_np,
            "counter_trans": np.array(counter_trans, dtype=np.float32),
            "total_count": np.array(total_count, dtype=np.float32),
            "rob_traj": robot_traj,
            "ped_traj": ped_traj,
            "pc_cur": pc_small,     
            "pc_next": pc_next,              
            "acc_x": np.array(max(acc_x), dtype=np.float32),
            "acc_y": np.array(max(acc_y), dtype=np.float32),                                  
        }
        rb_publisher.publish(tensor_dict_convert.to_ros_msg(batch_data))

        xhist7_x = xhist6_x
        xhist6_x = xhist5_x
        xhist5_x = xhist4_x
        xhist4_x = xhist3_x
        xhist3_x = xhist2_x
        xhist2_x = xhist1_x
        xhist1_x = xcgx
        xgoal_x = xpgx
        #current_obs_x = image_cat

        vel_previous = vel_x
        vel_x = np.array([msg_pub.linear.x, msg_pub.angular.z], dtype=np.float32)
        
        vel_vec[0,0] = msg_pub.linear.x
        vel_vec[0,1] = msg_pub.angular.z
        #vel_vec = vwres.cpu().numpy()
        
        cur_trans_mat_np = trans_mat_np
        cur_t265_mat_np = t265_mat_np
        
        pc_small = pc_next
        """
        """
        batch_data = {
            "observations": cur_obs,
            "actions": actions,
            "next_observations": next_obs,
            "trans_mat": trans_mat_np,     
            "t265_odom": t265_mat_np
            "goal_pose": gpose_mat_np                   
            "rewards": np.array([rewards]),
            "dones": np.array([1.0 if (terminated or truncated) else 0.0]),
            "masks": np.array([0.0 if terminated else 1.0]),
            "wandb_logging": wandb_logs,
        }
        rb_publisher.publish(tensor_dict_convert.to_ros_msg(batch_data))
        """
def callback_sub360(msg):
    global cur_img_360
    cur_img_360 = preprocess_image(msg) #current image

def callback_ref360(msg):
    global goal_img_360
    goal_img_360 = imgmsg_to_numpy(msg) #subgoal image
    #print("360", goal_img_360.shape)

def callback_ref(msg):
    global goal_img, goal_img_topic
    goal_img_topic = msg
    goal_img = imgmsg_to_numpy(msg) #subgoal image

def callback_ref_fisheye(msg):
    global goal_img
    #print("ugoiteru??")
    goal_img = imgmsg_to_numpy_realsense(msg) #subgoal image
    #print("realsense", goal_img.shape)

def callback_ref_realsense(msg):
    global goal_img
    goal_img = imgmsg_to_numpy_realsense(msg) #subgoal image
    #print("realsense", goal_img.shape)

#def callback_fisheye_collection(msg_1):
def callback_spherical_collection(msg_1):
    global bag_file
    global fcount
    global em_stop
    global prev_em_stop
    
    #print(em_stop, em_stop2, prev_em_stop)
    if rosparam.get_param("/goal_arrival") == 1:
        print("close bagfile and make new bagfile")
        bag_file.close()
        fcount += 1
        #bag_file = rosbag.Bag(newpath + '/test_test.bag', 'w')
        bag_file = rosbag.Bag(newpath + '/' + str(fcount).zfill(8) + '.bag', 'w')
        rospy.set_param("/goal_arrival", 0)

    if int(em_stop) == 1 and int(em_stop2) == 1:
        #bag_file.write('/fisheye_image/compressed', msg_1)
        bag_file.write('/spherical_image/compressed', msg_1)
        bag_file.write('/odometery', topic_odom)
        bag_file.write('/laserscan', topic_laserscan)
        #bag_file.write('/armarker', topic_armarker)
        bag_file.write('/bumper_roomba', topic_bumper)
        bag_file.write('/velocity_command', topic_velcmd)
        bag_file.write('/goal_img', goal_img_topic)
        bag_file.write('/goal_id', topic_goalID)        
        bag_file.write('/robot_traj', markerArray)
        bag_file.write('/odom_t265', topic_t265)     
        bag_file.write('/odom_trans', topic_odomt)             
    elif int(em_stop) == 0 and int(prev_em_stop) == 1 and int(em_stop2) == 1:
        #bag_file.write('/fisheye_image/compressed', msg_1)
        bag_file.write('/spherical_image/compressed', msg_1)
        bag_file.write('/odometery', topic_odom)
        bag_file.write('/laserscan', topic_laserscan)
        #bag_file.write('/armarker', topic_armarker)
        bag_file.write('/bumper_roomba', topic_bumper)
        bag_file.write('/velocity_command', topic_velcmd)
        bag_file.write('/goal_img', goal_img_topic)
        bag_file.write('/goal_id', topic_goalID)            
        bag_file.write('/robot_traj', markerArray)
        bag_file.write('/odom_t265', topic_t265) 
        bag_file.write('/odom_trans', topic_odomt)                  
    else:
        print("Collision. Not saving data!!")

    prev_em_stop = em_stop
    #global cv2_fisheye, topic_fisheye
    #topic_fisheye = msg_1
    #cv2_fisheye = bridge.imgmsg_to_cv2(msg_1, 'bgr8')

#def callback_spherical_collection(msg_1):
#    global topic_spherical
#    topic_spherical = msg_1
#    #bag_file.write('/spherical_image', msg_1)

def callback_laserscan_collection(msg_1):
    global topic_laserscan
    topic_laserscan = msg_1
    #bag_file.write('/spherical_image', msg_1)

#def callback_armarker_collection(msg_1):
#    global topic_armarker
#    topic_armarker = msg_1

def callback_goalID(msg_1):
    global topic_goalID
    topic_goalID = msg_1

def callback_t265(msg_1):
    global topic_t265
    topic_t265 = msg_1

def callback_t265_acc(msg_1):
    global acc_x, acc_y
    
    acc_x[1:20] = acc_x[0:19]
    acc_y[1:20] = acc_y[0:19]
    
    acc_x[0] = abs(msg_1.linear_acceleration.x)
    acc_y[0] = abs(msg_1.linear_acceleration.y)
    
    #print("X", acc_x)
    #print("Y", acc_y)
        
def callback_odomt(msg_1):
    global topic_odomt
    topic_odomt = msg_1

def callback_trans(msg_1):
    global topic_trans_mat
    topic_trans_mat = msg_1

def callback_goalpose(msg_1):
    global topic_gpose_mat
    topic_gpose_mat = msg_1
"""
def callback_pedest_org(msg):
    cv_img = bridge.imgmsg_to_cv2(msg)
    cv_trans = cv_img.transpose(2, 0, 1)
    cv_trans_np = np.array([cv_trans], dtype=np.float32)
    #ni, hi, wi = cv_trans_np.shape
    #cv_f = cv_trans_np[:, :, :, 0:int(wi/2)]
    #cv_b = cv_trans_np[:, :, :, int(wi/2):wi]

    #cv_fb = np.concatenate((cv_f, cv_b), axis=1)            
    double_fisheye_pano = torch.from_numpy(cv_trans_np).float().cuda()

    image_d = double_fisheye_pano
    #print(image_d.size())
    #print(mask_gpu.size(), transform(image_d[:,:,:,0:2*xyoffset]).size())
    image_d = torch.cat((mask_gpu*transform(image_d[:,:,:,0:2*xyoffset]).clone(), mask_gpu*transform(image_d[:,:,:,2*xyoffset:4*xyoffset]).clone()), dim=0)/255.0
    image_d_flip = torch.flip(image_d, [3])
    image_dc = torch.cat((image_d, image_d_flip), dim=0)
    #print(image_dc.size())
    with torch.no_grad():
        features = enc_depth(image_dc)   
        outputs_c, camera_param_c, binwidth_c, camera_range_c, camera_offset_c = dec_depth(features)                 
        
        outputs = (outputs_c[("disp", 0)][0:2,:,:,:] + torch.flip(outputs_c[("disp", 0)][2:4],[3]))*0.5
        camera_param = (camera_param_c[0:2] + camera_param_c[2:4])*0.5
        binwidth = (binwidth_c[0:2] + binwidth_c[2:4])*0.5
        camera_range = (camera_range_c[0:2] + camera_range_c[2:4])*0.5                                        
        camera_offset = (camera_offset_c[0:2] + camera_offset_c[2:4])*0.5

    cam_lens_x = []
    bdepth, _, _, _ = image_d.size()
    lens_zero = torch.zeros((bdepth, 1)).to("cuda")
    binwidth_zero = torch.zeros((bdepth, 1)).to("cuda")
    for i in range(16):
        lens_height = torch.zeros(bdepth, 1, device="cuda")
        for j in range(0, i+1):
            lens_height += camera_param[:, j:j+1]
        cam_lens_x.append(lens_height)
    cam_lens_c = torch.cat(cam_lens_x, dim=1)
    cam_lens = 1.0 - torch.cat([lens_zero, cam_lens_c], dim=1)

    lens_bincenter_x = []
    for i in range(16):
        bin_center = torch.zeros(bdepth, 1, device="cuda")
        for j in range(0, i+1):
            bin_center += binwidth[:, j:j+1]
        lens_bincenter_x.append(bin_center)
    lens_bincenter_c = torch.cat(lens_bincenter_x, dim=1)
    lens_bincenter = torch.cat([binwidth_zero, lens_bincenter_c], dim=1)
                
    lens_alpha = (cam_lens[:,1:16+1] - cam_lens[:,0:16])/(lens_bincenter[:,1:16+1] - lens_bincenter[:,0:16] + 1e-7)
    lens_beta = (-cam_lens[:,1:16+1]*lens_bincenter[:,0:16] + cam_lens[:,0:16]*lens_bincenter[:,1:16+1] + 1e-7)/(lens_bincenter[:,1:16+1] - lens_bincenter[:,0:16] + 1e-7)                
                
    double_disp = torch.cat((outputs[0:1], outputs[1:2]), dim=3)
    pred_disp_pano, pred_depth = disp_to_depth(double_disp, 0.1, 100.0)
                
    cam_points_f, h_back1, x_back1 = backproject_depth_fisheye[0](pred_depth[:,:,:,0:416], lens_alpha[0:1], lens_beta[0:1], camera_range[0:1], camera_offset[0:1])
    cam_points_b, h_back2, x_back2 = backproject_depth_fisheye[0](pred_depth[:,:,:,416:2*416], lens_alpha[1:2], lens_beta[1:2], camera_range[1:2], camera_offset[1:2])
                
    cam_points_fb16 = np.clip(torch.cat((cam_points_f, cam_points_b), axis=1).cpu().numpy(), -30.0, 30.0).astype(np.float16)

    ### Panorama image, depth and point clouds ###              
    panorama = pano_image.forward(double_fisheye_pano)
    panorama_disp = pano_image.forward(transform_raw(pred_disp_pano))

    double_x = torch.cat((cam_points_f[:,0:1].reshape(1,1,128,416), -cam_points_b[:,0:1].reshape(1,1,128,416)), dim=3)
    double_y = torch.cat((cam_points_f[:,1:2].reshape(1,1,128,416), cam_points_b[:,1:2].reshape(1,1,128,416)), dim=3)
    double_z = torch.cat((cam_points_f[:,2:3].reshape(1,1,128,416), -cam_points_b[:,2:3].reshape(1,1,128,416)), dim=3)
                
    panorama_x = pano_image.forward(transform_raw(double_x))
    panorama_y = pano_image.forward(transform_raw(double_y))
    panorama_z = pano_image.forward(transform_raw(double_z))

    ### peds. estimation ###
    img0 = panorama.squeeze(0).cpu().numpy().astype(np.uint8).transpose(1, 2, 0)[:, :, ::-1]               
    img = letterbox(img0, new_shape=1280)[0] #640

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
                
    with torch.no_grad():
        pred = detector(img, augment=True)[0]  # list: bz * [ (#obj, 6)]

    # Apply NMS and filter object other than person (cls:0)
    pred = non_max_suppression(pred, 0.7, 0.5, classes=[0], agnostic=True)

    # get all obj ************************************************************
    det = pred[0]  # for video, bz is 1
    if det is not None and len(det):  # det: (#obj, 6)  x1 y1 x2 y2 conf cls
        # Rescale boxes from img_size to original im0 size
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

        # Print results. statistics of number of each obj
        for c in det[:, -1].unique():
            n = (det[:, -1] == c).sum()  # detections per class
            s += '%g %ss, ' % (n, detector.names[int(c)])  # add to string

        bbox_xywh = xyxy2xywh(det[:, :4]).cpu()
        confs = det[:, 4:5].cpu()

        # ****************************** deepsort ****************************
        outputs = deepsort.update(bbox_xywh, confs, img0)
        # (#ID, 5) x1,y1,x2,y2,track_ID
    else:
        outputs = torch.zeros((0, 5))
                    
    last_out = outputs

    xpos_close = []
    zpos_close = []
    #if len(outputs) > 0: 
    if det is not None:
        min_dist = 100000000000                    
        #for ped in range(len(outputs)):
        for ped in range(len(bbox_xyxy[:])):
            center_u = (bbox_xyxy[ped][1] + bbox_xyxy[ped][3])*0.5
            center_v = (bbox_xyxy[ped][0] + bbox_xyxy[ped][2])*0.5
            width_u = bbox_xyxy[ped][3]-bbox_xyxy[ped][1]
            width_v = bbox_xyxy[ped][2]-bbox_xyxy[ped][0]
            us = int(center_u - width_u*0.125)
            ue = int(center_u + width_u*0.125)
            vs = int(center_v - width_v*0.125)
            ve = int(center_v + width_v*0.125)
            ped_pixels_x = panorama_x.squeeze(0).cpu().numpy()[:, us:ue, vs:ve]
            ped_pixels_y = panorama_y.squeeze(0).cpu().numpy()[:, us:ue, vs:ve]
            ped_pixels_z = panorama_z.squeeze(0).cpu().numpy()[:, us:ue, vs:ve]                                                                                
            xpos = np.median(ped_pixels_x)
            zpos = np.median(ped_pixels_z)  
            
            if xpos**2 + zpos**2 < min_dist:
                min_dist = xpos**2 + zpos**2
                xpos_close.append(xpos)
                zpos_close.append(zpos)
    
    print("peds", bbox_xyxy)
    print("xpos", xpos_close)
    print("zpos", zpos_close)  
"""    
def ped_transform(rob_traj, ped_traj):
    #for pedestrian and robot traj
    #rob_raw = self.buffer_loc[bl]["rob_traj"]
    #ped_raw = self.buffer_loc[bl]["ped_traj"]
    rob_raw = rob_traj
    ped_raw = ped_traj
    
    Trans_next = np.zeros((4, 4)) 
    Trans_next[0, 0] = np.cos(-rob_raw[2,0])
    Trans_next[0, 2] = np.sin(-rob_raw[2,0])
    Trans_next[1, 1] = 1.0
    Trans_next[2, 0] = -np.sin(-rob_raw[2,0])
    Trans_next[2, 2] = np.cos(-rob_raw[2,0])
    Trans_next[0, 3] = -rob_raw[1,0]
    Trans_next[2, 3] = rob_raw[0,0]
    Trans_next[3, 3] = 1.0                
    Tinv_next = np.linalg.inv(Trans_next)
    """       
    Trans_current = np.zeros((4, 4)) 
    Trans_current[0, 0] = np.cos(-rob_raw[2,1])
    Trans_current[0, 2] = np.sin(-rob_raw[2,1])
    Trans_current[1, 1] = 1.0
    Trans_current[2, 0] = -np.sin(-rob_raw[2,1])
    Trans_current[2, 2] = np.cos(-rob_raw[2,1])
    Trans_current[0, 3] = -rob_raw[1,1]
    Trans_current[2, 3] = rob_raw[0,1]
    Trans_current[3, 3] = 1.0              
    Tinv_current = np.linalg.inv(Trans_current)
    """                            
    Tr_c_list = []
    Tr_n_list = []
    Tp_c_list = []
    Tp_n_list = []                                         
    for i in range(8):
        Tr_n = np.zeros((4, 4)) 
        Tr_n[0, 0] = np.cos(-rob_raw[2,i])
        Tr_n[0, 2] = np.sin(-rob_raw[2,i])
        Tr_n[1, 1] = 1.0
        Tr_n[2, 0] = -np.sin(-rob_raw[2,i])
        Tr_n[2, 2] = np.cos(-rob_raw[2,i])
        Tr_n[0, 3] = -rob_raw[1,i]
        Tr_n[2, 3] = rob_raw[0,i]
        Tr_n[3, 3] = 1.0         
        """              
        Tr_c = np.zeros((4, 4)) 
        Tr_c[0, 0] = np.cos(-rob_raw[2,i+1])
        Tr_c[0, 2] = np.sin(-rob_raw[2,i+1])
        Tr_c[1, 1] = 1.0
        Tr_c[2, 0] = -np.sin(-rob_raw[2,i+1])
        Tr_c[2, 2] = np.cos(-rob_raw[2,i+1])
        Tr_c[0, 3] = -rob_raw[1,i+1]
        Tr_c[2, 3] = rob_raw[0,i+1]
        Tr_c[3, 3] = 1.0                             
        """
        Tp_n = np.zeros((4, 4)) 
        Tp_n[0, 0] = np.cos(0.0)
        Tp_n[0, 2] = np.sin(0.0)
        Tp_n[1, 1] = 1.0
        Tp_n[2, 0] = -np.sin(0.0)
        Tp_n[2, 2] = np.cos(0.0)
        Tp_n[0, 3] = ped_raw[0,i]
        Tp_n[2, 3] = ped_raw[1,i]
        Tp_n[3, 3] = 1.0         
        """              
        Tp_c = np.zeros((4, 4)) 
        Tp_c[0, 0] = np.cos(0.0)
        Tp_c[0, 2] = np.sin(0.0)
        Tp_c[1, 1] = 1.0
        Tp_c[2, 0] = -np.sin(0.0)
        Tp_c[2, 2] = np.cos(0.0)
        Tp_c[0, 3] = ped_raw[0,i+1]
        Tp_c[2, 3] = ped_raw[1,i+1]
        Tp_c[3, 3] = 1.0          
        """            
        if ped_raw[0,i] != 0.0 and ped_raw[1,i] != 0.0:    
            Tp_n_list.append(np.expand_dims(np.matmul(Tinv_next, np.matmul(Tr_n, Tp_n)), axis=0))
        else:
            Tp_n_list.append(np.expand_dims(np.eye(4), axis=0))
        """
        if ped_raw[0,i+1] != 0.0 and ped_raw[1,i+1] != 0.0:
            Tp_c_list.append(np.expand_dims(np.matmul(Tinv_current, np.matmul(Tr_c, Tp_c)), axis=0))
        else:
            Tp_c_list.append(np.expand_dims(np.eye(4), axis=0))                        
        """            
        Tr_n_list.append(np.expand_dims(np.matmul(Tinv_next, Tr_n), axis=0))
        #Tr_c_list.append(np.expand_dims(np.matmul(Tinv_current, Tr_c), axis=0))
                
    Tp_n_listcat = np.concatenate(Tp_n_list, axis=0)
    #Tp_c_listcat = np.concatenate(Tp_c_list, axis=0)
    Tr_n_listcat = np.concatenate(Tr_n_list, axis=0)
    #Tr_c_listcat = np.concatenate(Tr_c_list, axis=0)                                                
                
    ped_n_xy = np.concatenate((Tp_n_listcat[:,0,3], Tp_n_listcat[:,2,3]), axis=0)
    #ped_c_xy = np.concatenate((Tp_c_listcat[:,0,3], Tp_c_listcat[:,2,3]), axis=0)   
    robot_n_xy = np.concatenate((Tr_n_listcat[:,0,3], Tr_n_listcat[:,2,3]), axis=0)
    #robot_c_xy = np.concatenate((Tr_c_listcat[:,0,3], Tr_c_listcat[:,2,3]), axis=0)

    for i in range(6):
        if ped_n_xy[i+1] == 0 and ped_n_xy[i+1+8] == 0:
            if ped_n_xy[i] != 0 and ped_n_xy[i+2] != 0:
                ped_n_xy[i+1] = (ped_n_xy[i] + ped_n_xy[i+2])*0.5
                ped_n_xy[i+1+8] = (ped_n_xy[i+8] + ped_n_xy[i+2+8])*0.5                        
            if ped_n_xy[i+1] != 0 and ped_n_xy[i+1+8] != 0:
                if ped_n_xy[i] != 0 and ped_n_xy[i+2] != 0:
                    ped_n_xy[i+1] = (ped_n_xy[i] + ped_n_xy[i+1] + ped_n_xy[i+2])*0.3333
                    ped_n_xy[i+1+8] = (ped_n_xy[i+8] + ped_n_xy[i+1+8] + ped_n_xy[i+2+8])*0.3333
        """                
        if ped_c_xy[i+1] == 0 and ped_c_xy[i+1+8] == 0:
            if ped_c_xy[i] != 0 and ped_c_xy[i+2] != 0:
                ped_c_xy[i+1] = (ped_c_xy[i] + ped_c_xy[i+2])*0.5
                ped_c_xy[i+1+8] = (ped_c_xy[i+8] + ped_c_xy[i+2+8])*0.5
                #print("hokan A")                            
        if ped_c_xy[i+1] != 0 and ped_c_xy[i+1+8] != 0:
            if ped_c_xy[i] != 0 and ped_c_xy[i+2] != 0:
                ped_c_xy[i+1] = (ped_c_xy[i] + ped_c_xy[i+1] + ped_c_xy[i+2])*0.3333
                ped_c_xy[i+1+8] = (ped_c_xy[i+8] + ped_c_xy[i+1+8] + ped_c_xy[i+2+8])*0.3333                            
                #print("hokan B")
        """                
    if ped_n_xy[0] == 0 and ped_n_xy[8] == 0:
        if ped_n_xy[1] != 0 and ped_n_xy[2] != 0:
            delta_x = ped_n_xy[1] - ped_n_xy[2]
            delta_z = ped_n_xy[9] - ped_n_xy[10]       
            ped_n_xy[0] = delta_x + ped_n_xy[1]
            ped_n_xy[8] = delta_z + ped_n_xy[9]
                                  
    if ped_n_xy[7] == 0 and ped_n_xy[15] == 0:
        if ped_n_xy[6] != 0 and ped_n_xy[5] != 0:                
            delta_x = ped_n_xy[6] - ped_n_xy[5]
            delta_z = ped_n_xy[14] - ped_n_xy[13]       
            ped_n_xy[0] = delta_x + ped_n_xy[6]
            ped_n_xy[15] = delta_z + ped_n_xy[14]     

    """            
    if ped_c_xy[0] == 0 and ped_c_xy[8] == 0:
        if ped_c_xy[1] != 0 and ped_c_xy[2] != 0:
            delta_x = ped_c_xy[1] - ped_c_xy[2]
            delta_z = ped_c_xy[9] - ped_c_xy[10]       
            ped_c_xy[0] = delta_x + ped_c_xy[1]
            ped_c_xy[8] = delta_z + ped_c_xy[9]
                                  
    if ped_c_xy[7] == 0 and ped_c_xy[15] == 0:
        if ped_c_xy[6] != 0 and ped_c_xy[5] != 0:                
            delta_x = ped_c_xy[6] - ped_c_xy[5]
            delta_z = ped_c_xy[14] - ped_c_xy[13]       
            ped_c_xy[0] = delta_x + ped_c_xy[6]
            ped_c_xy[15] = delta_z + ped_c_xy[14]      
    """
    return ped_n_xy, robot_n_xy
def callback_pedest(msg):
    global xpos_close, zpos_close, pc_next
    
    #start = time.time()
    
    cv_img = bridge.imgmsg_to_cv2(msg)
    cv_trans = cv_img.transpose(2, 0, 1)
    cv_trans_np = np.array([cv_trans], dtype=np.float32)
    #ni, hi, wi = cv_trans_np.shape
    #cv_f = cv_trans_np[:, :, :, 0:int(wi/2)]
    #cv_b = cv_trans_np[:, :, :, int(wi/2):wi]

    #cv_fb = np.concatenate((cv_f, cv_b), axis=1)            
    double_fisheye_pano = torch.from_numpy(cv_trans_np).float().cuda()

    with_flip = False
    image_d = double_fisheye_pano
    image_d = torch.cat((mask_gpu*transform(image_d[:,:,:,0:2*xyoffset]).clone(), mask_gpu*transform(image_d[:,:,:,2*xyoffset:4*xyoffset]).clone()), dim=0)/255.0
    

    #cv_fb = np.concatenate((cv_f, cv_b), axis=1)            
    #double_fisheye_pano = torch.from_numpy(cv_trans_np).float().cuda()
    #print(transform(double_fisheye_pano[:,:,:,0:2*xyoffset]).size())
    #double_fisheye_raw = torch.cat((transform(double_fisheye_pano[:,:,:,0:2*xyoffset]).clone(), transform(double_fisheye_pano[:,:,:,2*xyoffset:4*xyoffset]).clone()), dim=3)

    #print("raw", double_fisheye_raw.size())
    #with_flip = False
    #image_d = torch.cat((mask_gpu*mask_gpu*double_fisheye_raw[:,:,:,0:416], mask_gpu*double_fisheye_raw[:,:,:,416:832]), dim=0)/255.0
    #image_d = double_fisheye_pano
    #image_d = torch.cat((mask_gpu*transform(image_d[:,:,:,0:2*xyoffset]).clone(), mask_gpu*transform(image_d[:,:,:,2*xyoffset:4*xyoffset]).clone()), dim=0)/255.0
    
    if with_flip:
        image_d_flip = torch.flip(image_d, [3])
        image_dc = torch.cat((image_d, image_d_flip), dim=0)
    else:
        image_dc = image_d
    #print(image_dc.size())
    
    #before_depth = time.time()     
    with torch.no_grad():
        #print(image_dc.size())
        #enc_depth_trt = torch2trt(enc_depth, [image_dc], fp16_mode=True)
        #features = enc_depth_trt(image_dc)          
        features = enc_depth(image_dc)   
        #dec_depth_trt = torch2trt(dec_depth, [features], fp16_mode=True)
        #outputs_c, camera_param_c, binwidth_c, camera_range_c, camera_offset_c = dec_depth_trt(features)          
        outputs_c, camera_param_c, binwidth_c, camera_range_c, camera_offset_c = dec_depth(features)   
        if with_flip:
            outputs_c, camera_param_c, binwidth_c, camera_range_c, camera_offset_c = dec_depth(features)   
            outputs = (outputs_c[("disp", 0)][0:2,:,:,:] + torch.flip(outputs_c[("disp", 0)][2:4],[3]))*0.5
            camera_param = (camera_param_c[0:2] + camera_param_c[2:4])*0.5
            binwidth = (binwidth_c[0:2] + binwidth_c[2:4])*0.5
            camera_range = (camera_range_c[0:2] + camera_range_c[2:4])*0.5                                        
            camera_offset = (camera_offset_c[0:2] + camera_offset_c[2:4])*0.5        
        else:
            outputs = outputs_c[("disp", 0)][0:2,:,:,:]
            camera_param = camera_param_c[0:2]
            binwidth = binwidth_c[0:2]
            camera_range = camera_range_c[0:2]                                   
            camera_offset = camera_offset_c[0:2]


    #after_depth = time.time()   
                
    #print(outputs.size())
    
    #cam_lens_x = []
    #bdepth, _, _, _ = image_d.size()
    #print(bdepth, bdepth)
    #lens_zero = torch.zeros((bdepth, 1)).to("cuda")
    #binwidth_zero = torch.zeros((bdepth, 1)).to("cuda")    
    """
    for i in range(16):
        lens_height = torch.zeros(bdepth, 1, device="cuda")
        for j in range(0, i+1):
            lens_height += camera_param[:, j:j+1]
        cam_lens_x.append(lens_height)
    cam_lens_c = torch.cat(cam_lens_x, dim=1)
    """
    cam_lens_c = torch.cumsum(camera_param[:, :], dim=1)    
    cam_lens = 1.0 - torch.cat([lens_zero, cam_lens_c], dim=1)

    #print(cam_lens_c)
    #print(cam_lens_cx)
    """
    lens_bincenter_x = []
    for i in range(16):
        bin_center = torch.zeros(bdepth, 1, device="cuda")
        for j in range(0, i+1):
            bin_center += binwidth[:, j:j+1]
        lens_bincenter_x.append(bin_center)
    lens_bincenter_c = torch.cat(lens_bincenter_x, dim=1)
    lens_bincenter = torch.cat([binwidth_zero, lens_bincenter_c], dim=1)
    """
    lens_bincenter_c = torch.cumsum(binwidth[:, :], dim=1)  
    lens_bincenter = torch.cat([binwidth_zero, lens_bincenter_c], dim=1)    
    #print(lens_bincenter_c)     
    #print(lens_bincenter_cx)  
                
    lens_alpha = (cam_lens[:,1:16+1] - cam_lens[:,0:16])/(lens_bincenter[:,1:16+1] - lens_bincenter[:,0:16] + 1e-7)
    lens_beta = (-cam_lens[:,1:16+1]*lens_bincenter[:,0:16] + cam_lens[:,0:16]*lens_bincenter[:,1:16+1] + 1e-7)/(lens_bincenter[:,1:16+1] - lens_bincenter[:,0:16] + 1e-7)                
                
    #double_disp = torch.cat((outputs[0:1], outputs[1:2]), dim=3)
    #
    #pred_disp_pano, pred_depth = disp_to_depth(double_disp, 0.1, 100.0)
    #            
    #cam_points_f, h_back1, x_back1 = backproject_depth_fisheye[0](pred_depth[:,:,:,0:416], lens_alpha[0:1], lens_beta[0:1], camera_range[0:1], camera_offset[0:1])
    #cam_points_b, h_back2, x_back2 = backproject_depth_fisheye[0](pred_depth[:,:,:,416:2*416], lens_alpha[1:2], lens_beta[1:2], camera_range[1:2], camera_offset[1:2])
                
    #cam_points_fb16 = np.clip(torch.cat((cam_points_f, cam_points_b), axis=1).cpu().numpy(), -30.0, 30.0).astype(np.float16)
    
    #print("cam_size", lens_alpha.size(), lens_beta.size(), camera_range.size(), camera_offset.size())
    '''
    lens_alpha = torch.ones((2, 16)).cuda()
    lens_beta = torch.ones((2, 16)).cuda()
    camera_range = torch.ones((2, 2)).cuda()    
    camera_offset = torch.ones((2, 2)).cuda()
    '''     
    double_disp = outputs
    pred_disp_pano, pred_depth = disp_to_depth(double_disp, 0.1, 100.0)
    cam_points_f_cat, h_back, x_back = backproject_depth_fisheye[0](pred_depth[:,:,:,0:416], lens_alpha[0:2], lens_beta[0:2], camera_range[0:2], camera_offset[0:2])
    cam_points_f = cam_points_f_cat[0:1]
    cam_points_b = cam_points_f_cat[1:2]
        
    #middle = time.time() 
    
    ### Panorama image, depth and point clouds ###    
             
    #panorama = pano_image.forward(double_fisheye_pano)
    #panorama_disp = pano_image.forward(transform_raw(pred_disp_pano))
 
    double_x = torch.cat((cam_points_f[:,0:1].reshape(1,1,128,416), -cam_points_b[:,0:1].reshape(1,1,128,416)), dim=3)
    double_y = torch.cat((cam_points_f[:,1:2].reshape(1,1,128,416), cam_points_b[:,1:2].reshape(1,1,128,416)), dim=3)
    double_z = torch.cat((cam_points_f[:,2:3].reshape(1,1,128,416), -cam_points_b[:,2:3].reshape(1,1,128,416)), dim=3)
    """            
    panorama_x = pano_image.forward(transform_raw(double_x))
    panorama_y = pano_image.forward(transform_raw(double_y))
    panorama_z = pano_image.forward(transform_raw(double_z))
    """    
    #print(double_fisheye_raw.size(), double_x.size())
    pc_est = torch.cat((double_x, double_y, double_z), dim=1)
    pano_cat = pano_image.forward(pc_est)
    panorama = pano_image.forward(double_fisheye_pano)
    panorama_x = pano_cat[:,0:1]
    panorama_y = pano_cat[:,1:2]
    panorama_z = pano_cat[:,2:3]
                    
    ### peds. estimation ###
    #print(double_fisheye_pano)
    #print(panorama.size())    
    """
    img0 = panorama.squeeze(0).cpu().numpy().astype(np.uint8).transpose(1, 2, 0)[:, :, ::-1]               
    img = letterbox(img0, new_shape=1280)[0] #640 1280

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    # numpy to tensor
    img = torch.from_numpy(img).to("cuda")
    
    #print(img.size())
    #print(panorama.size())
    """
    img = transform_yolo(panorama.squeeze(0))
    img = img.half()
    #img = img    
    img /= 255.0  # 0 - 255 to 0.0 - 1.0

    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    s = '%gx%g ' % img.shape[2:]    # print string
                
    #print(img.size())
    #before_yolo = time.time()     
    with torch.no_grad():
        #detector_trt = torch2trt(detector, [img], fp16_mode=True)
        #pred = detector_trt(img, augment=True)[0]         
        
        #print(img.size())
        pred = detector(img, augment=True)[0]  # list: bz * [ (#obj, 6)]


    #after_yolo = time.time()    
    # Apply NMS and filter object other than person (cls:0)
    
    #print("depth size", panorama_x.size())
    #print("RGB size", img.size())    
    
    pred = non_max_suppression(pred, 0.7, 0.5, classes=[0], agnostic=True)

    
    # get all obj ************************************************************
    det = pred[0]  # for video, bz is 1
    
    bbox_xyxy = []
    if det is not None and len(det):  # det: (#obj, 6)  x1 y1 x2 y2 conf cls
        # Rescale boxes from img_size to original im0 size
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], (560, 1120)).round()

        # Print results. statistics of number of each obj
        #for c in det[:, -1].unique():
        #    n = (det[:, -1] == c).sum()  # detections per class
        #    s += '%g %ss, ' % (n, detector.names[int(c)])  # add to string

        bbox_xyxy = det[:, :4]
    
    '''
    if det is not None:
        bbox_xyxy = det[:, :4]
    else:
        bbox_xyxy = []        
    '''
    """
        bbox_xywh = xyxy2xywh(det[:, :4]).cpu()
        confs = det[:, 4:5].cpu()

        # ****************************** deepsort ****************************
        outputs = deepsort.update(bbox_xywh, confs, img0)
        # (#ID, 5) x1,y1,x2,y2,track_ID
    else:
        outputs = torch.zeros((0, 5))
                    
    last_out = outputs
    """
    """
    #Boundary boxes
    #Boxes = []
    if len(outputs) > 0:                  
        #print(outputs, outputs[:, :4]) 
        bbox_xyxy = outputs[:, :4]
        identities = outputs[:, -1]
        #scale = 128/550
        #for ped in range(len(outputs)):
        #    boundingbox = BoundingBox(xmin = int(scale*outputs[ped][0]), xmax = int(scale*outputs[ped][2]), ymin = int(scale*outputs[ped][1]), ymax = int(scale*outputs[ped][3]), id = int(outputs[ped][4])) #, Class = object_name, probability = scores[i]
        #    Boxes.append(boundingbox)
    """
    xpos_close = []
    zpos_close = []
    #if len(outputs) > 0: 
    if det is not None:
        min_dist = 100000000000                    
        #for ped in range(len(outputs)):
        for ped in range(len(bbox_xyxy[:])):
            center_u = (bbox_xyxy[ped][1] + bbox_xyxy[ped][3])*0.5
            center_v = (bbox_xyxy[ped][0] + bbox_xyxy[ped][2])*0.5
            width_u = bbox_xyxy[ped][3]-bbox_xyxy[ped][1]
            width_v = bbox_xyxy[ped][2]-bbox_xyxy[ped][0]
            us = int(center_u - width_u*0.125)
            ue = int(center_u + width_u*0.125)
            vs = int(center_v - width_v*0.125)
            ve = int(center_v + width_v*0.125)
            #ped_pixels_x = panorama_x.squeeze(0).cpu().numpy()[:, us:ue, vs:ve]
            #ped_pixels_y = panorama_y.squeeze(0).cpu().numpy()[:, us:ue, vs:ve]
            #ped_pixels_z = panorama_z.squeeze(0).cpu().numpy()[:, us:ue, vs:ve]       
            ped_pixels_x = panorama_x.squeeze(0)[:, us:ue, vs:ve]
            ped_pixels_y = panorama_y.squeeze(0)[:, us:ue, vs:ve]
            ped_pixels_z = panorama_z.squeeze(0)[:, us:ue, vs:ve]                                                                                            
            xpos = torch.median(ped_pixels_x)
            zpos = torch.median(ped_pixels_z)  
            
            if xpos**2 + zpos**2 < min_dist:
                min_dist = xpos**2 + zpos**2
                xpos_close.append(xpos.cpu().numpy())
                zpos_close.append(zpos.cpu().numpy())
    
    end = time.time()
    
    pc_next = transform_depth(pc_est).cpu().squeeze(0).numpy()
    #print(pc_small.shape)
    #print("peds", bbox_xyxy)
    print("xpos", xpos_close)
    print("zpos", zpos_close)  
    #print(end-start, before_depth-start, after_depth-start, middle-start, before_yolo-start, after_yolo-start)
    #print(end-start)    
    """
    in_img1 = (in_imgcc1 - 127.5)/127.5

    img_nn_cL = mask_brrc * (in_img1 + 1.0) -1.0 #mask
    img = img_nn_cL.astype(np.float32)

    cv_resize_nx = np.concatenate((cv_resize_Fp, cv_resize_Bp), axis=1)            
    double_fisheye_pano = torch.from_numpy(cv_resize_nx.transpose(2,0,1)).float().cuda().unsqueeze(dim=0)
    """

isgoal = False
em_stop = 1.0
em_stop2 = 1.0
prev_em_stop = 1.0
left_bumper = 1.0
right_bumper = 1.0
count_comeback = 20

flag_slow = 1.0
left_bumper_light = 1.0
right_bumper_light = 1.0
left_light = 0.0
right_light = 0.0

count_collision = 0
flag_countup = 0
flag_send = 0

pc_small = np.zeros((3,128,256), dtype=np.float32)
pc_next = np.zeros((3,128,256), dtype=np.float32)

xpos_close = []
zpos_close = []

acc_x = [0]*20
acc_y = [0]*20

roll = 0.0
pitch = 0.0
yaw = 0.0
x_pos = 0.0
y_pos = 0.0
z_pos = 0.0

rob9_x = 0.0
rob8_x = 0.0
rob7_x = 0.0
rob6_x = 0.0
rob5_x = 0.0
rob4_x = 0.0
rob3_x = 0.0
rob2_x = 0.0
rob1_x = 0.0

rob9_y = 0.0
rob8_y = 0.0
rob7_y = 0.0
rob6_y = 0.0
rob5_y = 0.0
rob4_y = 0.0
rob3_y = 0.0
rob2_y = 0.0
rob1_y = 0.0

rob9_yaw = 0.0
rob8_yaw = 0.0
rob7_yaw = 0.0
rob6_yaw = 0.0
rob5_yaw = 0.0
rob4_yaw = 0.0
rob3_yaw = 0.0
rob2_yaw = 0.0
rob1_yaw = 0.0

ped9_x = 0.0
ped8_x = 0.0
ped7_x = 0.0
ped6_x = 0.0
ped5_x = 0.0
ped4_x = 0.0
ped3_x = 0.0
ped2_x = 0.0
ped1_x = 0.0

ped9_z = 0.0
ped8_z = 0.0
ped7_z = 0.0
ped6_z = 0.0
ped5_z = 0.0
ped4_z = 0.0
ped3_z = 0.0
ped2_z = 0.0
ped1_z = 0.0

#transform = T.Resize((320,320))

def callback_isgoal(msg):
    global isgoal
    isgoal = msg.data

def callback_bumper(msg):
    global em_stop, vjoy, wjoy
    global cur_img_raw
    global topic_bumper
    global left_bumper, right_bumper, count_comeback
    global flag_slow
    global left_light, right_light
    global flag_countup, count_collision
    global flag_send, em_stop2 
    sec_send = 10

    #print(flag_countup, count_collision, em_stop)
    topic_bumper = msg
    if msg.is_left_pressed or msg.is_right_pressed and flag_countup == 1:
        flag_countup = 1
        if int(em_stop) == 1 and count_collision < int(10*sec_send) and count_collision > int(10*5):
            #cv2.imwrite(newpath + '/tmp' + '/current.png',cur_img_raw)
            #sent = client.sendLocalImage(newpath + '/tmp' + '/current.png', message=msg_user, thread_id=friend.uid, thread_type=thread_type)
            #if sent:
            #    print("Message sent successfully!")
            em_stop = 0.0
        print(msg.is_left_pressed, msg.is_right_pressed)
    elif msg.is_left_pressed or msg.is_right_pressed:
        flag_countup = 1
    """
    if rosparam.get_param("/out_map") == 1 and flag_send == 0:
        cv2.imwrite(newpath + '/tmp' + '/current.png',cur_img_raw)
        sent = client.sendLocalImage(newpath + '/tmp' + '/current.png', message=msg_user, thread_id=friend.uid, thread_type=thread_type)
        flag_send = 1
    """ 
    #print("ROS PARAM", rosparam.get_param("/pose_err"), flag_send)
    #if rosparam.get_param("/pose_err") > 5.0 and flag_send == 0:
    if flag_send == 0:    
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")       
    
        with open(newpath + '/' + 'out_of_map.txt', 'a') as f:
            f.write(current_time)            
            f.write('\n')     

        flag_send = 1                  
        em_stop2 = 0
        
        #facebook chatbot
        #cv2.imwrite(newpath + '/tmp' + '/current.png',cur_img_raw)
        #sent = client.sendLocalImage(newpath + '/tmp' + '/current.png', message=msg_user2, thread_id=friend.uid, thread_type=thread_type)      
        
    #elif rosparam.get_param("/pose_err") < 3.0:
    #    em_stop2 = 1
    #    flag_send = 0
    
    if flag_countup == 1:
        count_collision += 1

    if count_collision > int(10*sec_send):
        flag_countup = 0
        count_collision = 0
    
    if not(msg.is_left_pressed) and not(msg.is_right_pressed):
        em_stop = 1.0
        
    if msg.is_left_pressed:
        left_bumper = 0
    if msg.is_right_pressed:
        right_bumper = 0
    if msg.is_left_pressed or msg.is_right_pressed:
        count_comeback = 0

    left_light = msg.light_signal_left + msg.light_signal_front_left + msg.light_signal_center_left
    right_light = msg.light_signal_right + msg.light_signal_front_right + msg.light_signal_center_right

    light_th = 100
    if msg.light_signal_left < light_th and msg.light_signal_front_left < light_th and msg.light_signal_center_left < light_th and msg.light_signal_center_right < light_th and msg.light_signal_front_right < light_th and msg.light_signal_right < light_th:
        flag_slow = 0
    else:
        flag_slow = 1

    #print("flag_slow", flag_slow, msg.light_signal_left, msg.light_signal_front_left, msg.light_signal_center_left, msg.light_signal_center_right, msg.light_signal_front_right, msg.light_signal_right)

def callback_joy(msg):
    global vjoy, wjoy
    vjoy = msg.linear.x
    wjoy = msg.angular.z
    """
    if vjoy < 0.00001 and wjoy < 0.00001:
        vjoy = 0.00001
        wjoy = 0.00001
    else:
        vjoy = msg.linear.x
        wjoy = msg.angular.z
    """
    #em_stop = 1.0

def receive_params(msg: tm.TensorDict):
    global polinet_actor
    print("parameter received and updated !!!")
    new_parameters = tensor_dict_convert.from_ros_msg(msg)
    for param_name, param in polinet_actor.state_dict().items():
        #print(param_name, param.sum())    
        #print(new_parameters[param_name])
        param.copy_(torch.from_numpy(new_parameters[param_name]).to(device))
    #for param_name, param in polinet.state_dict().items():
    #    print(param_name, param.sum())    
    
    polinet_actor.eval()
    print("parameter received and updated")
    ##
    #if self.agent is not None:
    #    params = tensor_dict_convert.from_ros_msg(msg)
    #    new_actor = self.agent.actor.replace(params=params)
    #    self.agent = self.agent.replace(actor=new_actor)

latest_image = None

def subscriber_callback(msg):
    global latest_image
    latest_image = msg
    
def timer_callback(_):
    global latest_image
    if latest_image is not None:
        callback_360(latest_image)
        latest_image = None    

#device = torch.device(﻿"cuda:0" if torch.cuda.is_available(﻿) else "cpu"﻿)
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

dof_size = 2
step_size = 8
lower_bounds = [0.0, -1.0]
upper_bounds = [+0.5, +1.0]

bridge = CvBridge()

#vwkeep = cuda.to_gpu(np.zeros((16,), dtype=np.float32))
vwkeep = torch.zeros((16,), dtype=torch.float32).to(device)
"""
if image360:
    cur_img_raw = cv2.imread("/home/vizbot/dumy_ref/360img/img_no1berkeley_27.jpg")
    goal_img_raw = cv2.imread("/home/vizbot/dumy_ref/360img/img_no1berkeley_39.jpg")
    cur_img_360 = cv2.imread("/home/vizbot/dumy_ref/360img/img_no1berkeley_27.jpg")
    goal_img_360 = cv2.imread("/home/vizbot/dumy_ref/360img/img_no1berkeley_39.jpg")
else:
    cur_img_rawx = cv2.imread("/home/vizbot/dumy_ref/realsense/img_no1berkeley_32.jpg")
    goal_img_rawx = cv2.imread("/home/vizbot/dumy_ref/realsense/img_no1berkeley_39.jpg")
    cur_img_raw = cv2.resize(cur_img_rawx, (rsizex, rsizey), cv2.INTER_AREA)
    goal_img_raw = cv2.resize(goal_img_rawx, (rsizex, rsizey), cv2.INTER_AREA)
    cur_img_360 = cv2.imread("/home/vizbot/dumy_ref/360img/img_no1berkeley_32.jpg")
    goal_img_360 = cv2.imread("/home/vizbot/dumy_ref/360img/img_no1berkeley_39.jpg")

cv_resizex = cur_img_raw.transpose(2, 0, 1)
in_imgcc1 = np.array([cv_resizex], dtype=np.float32)
in_img1 = (in_imgcc1 - 127.5)/127.5

if image360:
    img_nn_cL = mask_brrc * (in_img1 + 1.0) -1.0 #mask
else:
    img_nn_cL = (in_img1 + 1.0) -1.0 #mask

cur_img = img_nn_cL.astype(np.float32)

cv_resizex = goal_img_raw.transpose(2, 0, 1)
in_imgcc1 = np.array([cv_resizex], dtype=np.float32)
in_img1 = (in_imgcc1 - 127.5)/127.5

if image360:
    img_nn_cL = mask_brrc * (in_img1 + 1.0) -1.0 #mask
else:
    img_nn_cL = (in_img1 + 1.0) -1.0 #mask
goal_img = img_nn_cL.astype(np.float32)
#print(goal_img.shape)

cv_resizex = cur_img_360.transpose(2, 0, 1)
in_imgcc1 = np.array([cv_resizex], dtype=np.float32)
in_img1 = (in_imgcc1 - 127.5)/127.5

img_nn_cL = mask_brrc * (in_img1 + 1.0) -1.0 #mask
cur_img_360 = img_nn_cL.astype(np.float32)

cv_resizex = goal_img_360.transpose(2, 0, 1)
in_imgcc1 = np.array([cv_resizex], dtype=np.float32)
in_img1 = (in_imgcc1 - 127.5)/127.5

img_nn_cL = mask_brrc * (in_img1 + 1.0) -1.0 #mask
goal_img_360 = img_nn_cL.astype(np.float32)
"""
#batchsize = 1
if image360:
    goal_img = np.zeros((1,3,128,256), dtype=np.float32)
else:
    goal_img = np.zeros((1,3,128,128), dtype=np.float32)
    cur_img_360 = np.zeros((1,3,128,256), dtype=np.float32)
    goal_img_360 = np.zeros((1,3,128,256), dtype=np.float32)

countm = 0
for it in range(128):
    for jt in range(256):
        if mask_brr[0][0][it][jt] > 0.5:
            countm += 1

if image360 is False:
    countm = 128*128 

print(countm)
mask_c = np.concatenate((mask_brr, mask_brr, mask_brr), axis=1)

### for depth estimation ###
print("Define and Load depth models.")
xyoffset = 280
"""
enc_depth = networks.ResnetEncoder(18, True, num_input_images = 1)
path = os.path.join("/home/vizbot/DeepSORT_YOLOv5_Pytorch/model/depthest_ploss/", "encoder.pth")
model_dict = enc_depth.state_dict()

pretrained_dict = torch.load(path, map_location=torch.device('cpu'))
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
enc_depth.load_state_dict(model_dict)
enc_depth.eval().to("cuda")

#xx = torch.ones((2, 3, 128, 416)).cuda()
#enc_depth_trt = torch2trt(enc_depth, [xx], fp16_mode=True)

dec_depth = networks.DepthDecoder_camera_ada4(enc_depth.num_ch_enc, [0, 1, 2, 3], 16)
path = os.path.join("/home/vizbot/DeepSORT_YOLOv5_Pytorch/model/depthest_ploss/", "depth.pth")
model_dict = dec_depth.state_dict()

pretrained_dict = torch.load(path, map_location=torch.device('cpu'))
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
dec_depth.load_state_dict(model_dict)
dec_depth.eval().to("cuda")

backproject_depth_fisheye = {}
project_3d_fisheye = {}
for scale in [0, 1, 2, 3]:
    h = 128 // (2 ** scale)
    w = 416 // (2 ** scale)
    backproject_depth_fisheye[scale] = BackprojectDepth_fisheye_inter_offset(2, h, w, 16)
    backproject_depth_fisheye[scale].to("cuda")
    project_3d_fisheye[scale] = Project3D_fisheye_inter_offset(1, h, w, 16)
    project_3d_fisheye[scale].to("cuda")

#lens_parameters
lens_zero = torch.zeros((2, 1)).to("cuda")
binwidth_zero = torch.zeros((2, 1)).to("cuda")    
"""
#Project3D_fisheye_inter_footprint(1, 128, 416, 16)    
transform = T.Resize(size = (128,416))
transform_depth = T.Resize(size = (128,256))
transform_raw = T.Resize(size = (2*xyoffset,4*xyoffset))
transform_yolo = T.Resize(size = (640, 1280))

pano_image = Pano_image(2*xyoffset, 4*xyoffset, 1)

#xy_subgoal = [[10.0, 0.0]]   
#xy_subgoal = [[6.0, -10.0]]   
#xy_subgoal = [[10.0, 0.0]]   
#xy_subgoal = [[10.0, 5.0]]   
#xy_subgoal = [[10.0, 0.0]]  
#xy_subgoal = [[3.0, 10.0]]  
#xy_subgoal = [[10.0, 10.0]]  
#xy_subgoal = [[10.0, 10.0]]
#xy_subgoal = [[20.0, 20.0]]
xy_subgoal = [[10.0, 5.0]]
#xy_subgoal = [[11.0, 2.0],[8.0, -10.0]]    
#xy_subgoal = [[6.0, 0.5],[4.0, 10.0]]    
#xy_subgoal = [[10.0, 0.0],[25.0, -3.0]]  
#xy_subgoal = [[10.0, 0.0]]    
yaw_ang1 = 90/180*3.14
yaw_ang2 = 90/180*3.14
yaw_ang3 = 0/180*3.14
yaw_ang4 = 90/180*3.14
yaw_ang5 = 0/180*3.14
yaw_ang6 = 0/180*3.14
id_goal = 0
#yaw_subgoal = [0.0/180*3.14]   
#yaw_subgoal = [-90.0/180*3.14]     
#yaw_subgoal = [0.0/180*3.14] 
yaw_subgoal = [90.0/180*3.14]  
"""
# ***************************** initialize DeepSORT **********************************
cfg = get_config()
cfg.merge_from_file("/home/vizbot/DeepSORT_YOLOv5_Pytorch/configs/deep_sort.yaml")

device_tr = select_device("")
#print(device_tr)
use_cuda = device_tr.type != 'cpu' and torch.cuda.is_available()
deepsort = build_tracker(cfg, use_cuda=use_cuda)
#device_tr = 'cpu'
#detector = torch.load('./mnt/yolov5/weights/yolov5s.pt', map_location=device_tr)['model'].float()  # load to FP32
detector = torch.load('/home/vizbot/DeepSORT_YOLOv5_Pytorch/yolov5/weights/yolov5s.pt', map_location=device_tr)['model'].float()  # load to FP32

detector.to(device_tr).eval()
half = device_tr.type != 'cpu'
if half:
    detector.half()  # to FP16

names = detector.module.names if hasattr(detector, 'module') else detector.names

#xx = torch.ones((1, 3, 640, 1280)).cuda()
#detector_trt = torch2trt(detector, [xx], fp16_mode=True)

# main function

# ***************************** initialize YOLO-V5 **********************************
"""
"""
#ped model
pednet = PedNet_delta_small(8, 8).eval().to("cuda")
pednet_fn = os.path.join("/home/vizbot/vizbot/src/p/dvmpc_pytorch/models", "polinet_state_lattice.pth") #polinet_state_lattice.pth
pednet.load_state_dict(unddp_state_dict(torch.load(pednet_fn, map_location=device_tr)))
print(device_tr)
"""
if __name__ == '__main__':
    #dummy
    #rospy.set_param("/goal_arrival", 0)
    counter_trans = 1
    total_count = 1
    
    v_prev = 0.0
    w_prev = 0.0
    #initialize node
    rospy.init_node('DVMPC', anonymous=False)

    callback_state_lattice()

    #print('sleeping 10s')
    #rospy.sleep(10.0)
    #subscribe of topics
    #msg1_sub = rospy.Subscriber('/cv_camera_node/image_raw', Image, callback, queue_size=1)
    if image360:
        #msg1_sub = rospy.Subscriber('/image_processed2', Image, callback_360, queue_size=1)
        #msg1_sub = rospy.Subscriber('/usb_cam/image_raw', Image, callback_fisheye, queue_size=1)
        #rospy.Subscriber('/image_processed2', Image, subscriber_callback)
        rospy.Subscriber('/usb_cam/image_raw', Image, subscriber_callback)        
        rospy.Timer(rospy.Duration(0.1), timer_callback)
    
        msg2_sub = rospy.Subscriber('/topoplan/image_ref', Image, callback_ref, queue_size=1)
    elif fisheye:
        #print("kiteruyone??")
        msg1_sub = rospy.Subscriber('/usb_cam/image_raw', Image, callback_fisheye, queue_size=1)
        msg2_sub = rospy.Subscriber('/topoplan/image_ref', Image, callback_ref_fisheye, queue_size=1)
    elif rsense:
        msg1_sub = rospy.Subscriber('/camera/color/image_raw_throttle', Image, callback_rsense, queue_size=1)
        msg2_sub = rospy.Subscriber('/topoplan/image_ref', Image, callback_ref_realsense, queue_size=1)

        #msg1_sub = rospy.Subscriber('/camera/color/image_raw_throttle', Image, callback_gen, queue_size=1)
        #msg2_sub = rospy.Subscriber('/topoplan/image_ref', Image, callback_ref_realsense, queue_size=1)
        #msg_sub = rospy.Subscriber('/image_processed', Image, callback_sub360, queue_size=1)
        #msg_sub2 = rospy.Subscriber('/topoplan/image_ref360', Image, callback_ref360, queue_size=1)

    msg3_sub = rospy.Subscriber('/topoplan/isgoal', Bool, callback_isgoal, queue_size=1)
    msg5_sub = rospy.Subscriber('/cv_camera_node/image_raw/compressed_throttle', CompressedImage, callback_spherical_collection, queue_size=1)
    msg6_sub = rospy.Subscriber('/scan', LaserScan, callback_laserscan_collection, queue_size=1)
    #msg7_sub = rospy.Subscriber('/ar_pose_marker2', AlvarMarkers, callback_armarker_collection, queue_size = 1)
    msg_bumper = rospy.Subscriber('/bumper', Bumper, callback_bumper, queue_size=1)
    msg_joy = rospy.Subscriber('/cmd_vel_ac', Twist, callback_joy, queue_size=1)
    
    sub = rospy.Subscriber ('/odom', Odometry, get_rotation)
    sub_goalID = rospy.Subscriber ('/topoplan/goalID', Int32, callback_goalID, queue_size=1)
    sub_t265 = rospy.Subscriber('/camera/odom/sample_throttle', Odometry, callback_t265, queue_size = 1)
    sub_t265_acc = rospy.Subscriber('/camera/accel/sample', Imu, callback_t265_acc, queue_size = 1)    
    sub_odomt = rospy.Subscriber('/odometry_trans', Odometry, callback_odomt, queue_size = 1)
    sub_trans = rospy.Subscriber('/pose_trans', Pose, callback_trans, queue_size = 1)
    sub_goalpose = rospy.Subscriber('/pose_goal', Pose, callback_goalpose, queue_size = 1)
    param_subscription = rospy.Subscriber(rospy.get_param("~param_topic", "/actor_params"), tm.TensorDict, receive_params, queue_size = 1)

    #publisher of topics
    msg_out = rospy.Publisher('/cmd_vel', Twist,queue_size=1) #velocities for the robot control
    msg_out_raw = rospy.Publisher('/cmd_vel_raw', Twist,queue_size=1)
    crop_image = rospy.Publisher('/crop_image',Image,queue_size=1)   #reference images
    image_pred = rospy.Publisher('/topoplan/image_pred',Image,queue_size=1)   #reference images
    image_ref_input = rospy.Publisher('/topoplan/image_ref_input',Image,queue_size=1)   #reference images
    image_cur_input = rospy.Publisher('/topoplan/image_cur_input',Image,queue_size=1)   #reference images
    array_robot_traj = rospy.Publisher('array_robot_traj', MarkerArray,queue_size=1)

    rb_publisher = rospy.Publisher(rospy.get_param("~rb_topic", "/replay_buffer_data"), tm.TensorDict, queue_size=100,)

    TrainingRosInterface()

    print('waiting message .....')
    rospy.spin()
