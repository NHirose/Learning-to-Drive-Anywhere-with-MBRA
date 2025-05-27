#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '../train')

import os
import time

from scipy.spatial.transform import Rotation as R

#ROS
import rospy
from geometry_msgs.msg import Twist, Pose
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry

#torch
import torch
import torch.nn.functional as F
import yaml

#others
from cv_bridge import CvBridge
from PIL import Image as PILImage
import numpy as np
import math
import transforms3d
from transforms3d import quaternions
from utils_logonav import load_model, msg_to_pil, transform_images_mbra, to_numpy

# load model weights
model_config_path = "../train/config/LogoNav.yaml"
with open(model_config_path, "r") as f:
    model_params = yaml.safe_load(f)
    
#checkpoint path    
ckpth_path = "./model_weights/logonav.pth"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

def msg_to_pil(msg: Image) -> PILImage.Image:
    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
        msg.height, msg.width, -1)
    pil_image = PILImage.fromarray(img)
    return pil_image


def pil_to_msg(pil_img: PILImage.Image, encoding="mono8") -> Image:
    img = np.asarray(pil_img)  
    ros_image = Image(encoding=encoding)
    ros_image.height, ros_image.width, _ = img.shape
    ros_image.data = img.ravel().tobytes() 
    ros_image.step = ros_image.width
    return ros_image
   
def calc_relative_pose(pose_a, pose_b):
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

def callback_logonav(msg_1):
    global init_hist, store_hist, image_hist
    global id_goal

    if True:
        newsize = model_params["image_size"]   
        context_size = model_params["context_size"]
        im = msg_to_pil(msg_1)
        #im_crop_o = im.crop((50, 100, 1230, 860)) #you may need to crop your raw camera image to be better performance.
        im_crop_o = im
        im_crop = im_crop_o.resize(newsize, PILImage.Resampling.LANCZOS).convert('RGB')     

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

        obs_images = transform_images_mbra(image_hist)
        obs_images = torch.split(obs_images, 3, dim=1)
        obs_images = torch.cat(obs_images, dim=1) 
        batch_obs_images = obs_images.to(device)
          
        #Current robot pose on the global coordinate       
        x_position = xxxxxxxxxxxx
        y_position = xxxxxxxxxxxx                
        yaw_angle = xxxxxxxxxxxx
        
        with torch.no_grad():                        
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
                              
                if np.sqrt(relative_x**2 + relative_y**2) < thres_update and id_goal != len(yaw_subgoal) - 1:              
                    id_goal += 1                     
                    
            else:
                goal_pose = np.array([100.0, 0.0, 1.0, 0.0])            
            
            goal_pose_torch = torch.from_numpy(goal_pose).unsqueeze(0).float().to(device)
            
            print("robot pose", x_position, y_position, yaw_angle)
            print("relative pose", goal_pose[0]*metric_waypoint_spacing, goal_pose[1]*metric_waypoint_spacing, goal_pose[2], goal_pose[3], id_goal)            
            with torch.no_grad():  
                waypoints = model(batch_obs_images, goal_pose_torch)         
            waypoints = to_numpy(waypoints)
        
        #PD controller from ViNT and NoMaD    
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

        maxv = 0.2
        maxw = 0.2        

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

        msg_out.publish(msg_pub)
        print("linear vel", msg_pub.linear.x, "angular_vel", msg_pub.angular.z)             

latest_image = None
def subscriber_callback(msg):
    global latest_image
    latest_image = msg
    
def timer_callback(_):
    global latest_image
    if latest_image is not None:
        callback_logonav(latest_image)
        latest_image = None    

#goal pose on the global coordinate (you can give list to have subgoals.)
xy_subgoal = [[8.0, -10.0]]
yaw_subgoal = [-90.0/180*3.14]  

if __name__ == '__main__':
    id_goal = 0
    store_hist = 0
    init_hist = 0
    image_hist = []
    
    bridge = CvBridge()
    rospy.init_node('LogoNav', anonymous=False)
    
    #subscriber of topics
    rospy.Subscriber('/usb_cam/image_raw', Image, subscriber_callback)        
    rospy.Timer(rospy.Duration(0.1), timer_callback)

    #publisher of topics
    msg_out = rospy.Publisher('/cmd_vel', Twist,queue_size=1) #velocities for the robot control

    print('waiting message .....')
    rospy.spin()
