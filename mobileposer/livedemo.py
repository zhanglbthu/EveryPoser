import torch
from pygame.time import Clock

import articulate as art
import os
from config import *
from articulate.utils.noitom import *
from articulate.utils.unity import MotionViewer

from articulate.utils.pygame import StreamingDataViewer
from articulate.utils.bullet.view_rotation_np import RotationViewer
from auxiliary import calibrate_q, quaternion_inverse
from utils.model_utils import load_model
import numpy as np
import matplotlib
from argparse import ArgumentParser
import keyboard
from apple_sensor.sensor import AppleSensor, CalibratedAppleSensor
from scipy.spatial.transform import Rotation as R
from articulate.utils.noitom.PN_lab import CalibratedIMUSet
import keyboard
import traceback

colors = matplotlib.colormaps['tab10'].colors
body_model = art.ParametricModel(paths.smpl_file, device='cuda')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--name', type=str, default='default')
    parser.add_argument('--noitom', action='store_true', help='use noitom imu')
    parser.add_argument('--apple', action='store_true', help='use apple imu')
    args = parser.parse_args()
    
    device = torch.device("cuda")
    clock = Clock()
    
    # set mobileposer network
    ckpt_path = "data/checkpoints/weights.pth"
    net = load_model(ckpt_path)
    net.eval()
    print('Mobileposer model loaded.')
    combo = [0, 3, 4]

    if args.noitom:
        # add ground truth readings
        imu_set = CalibratedIMUSet()
        imu_set.calibrate('walking_9dof')
    
    if args.apple:
        apple_sensor = CalibratedAppleSensor(AppleDevices.udp_ports, AppleDevices.device_ids)
        apple_sensor.calibrate("walking_6dof")
    
    accs, oris, poses, trans = [], [], [], []
    
    idx = 0
    # sviewer = StreamingDataViewer(3, y_range=(-10, 10), window_length=200, names=['X', 'Y', 'Z']); sviewer.connect()
    # rviewer = RotationViewer(2, order='wxyz'); rviewer.connect()
    with torch.no_grad(), MotionViewer(1, overlap=False, names=['apple_mocap']) as viewer:
        while True:
            try:
                clock.tick(30)
                viewer.clear_all(render=False)
                ori = torch.zeros(5, 3, 3).to(device)
                a = torch.zeros(5, 3).to(device)
                if args.noitom:
                    # gt readings
                    _, RIS, aS, wS, mS, aI, wI, mI, RMB, aM, wM, mM = imu_set.get()
                    
                    ori[combo] = RMB[combo].to(device)
                    a[combo] = aM[combo].to(device)
                    
                if args.apple:
                    # device readings
                    t, aS, aI, aM, RIS, RMB = apple_sensor.get()
                    ori[combo] = RMB[[1, 0, 2]].to(device)
                    a[combo] = aM[[1, 0, 2]].to(device)
                oris.append(ori)
                accs.append(a)

                ori = ori.view(5, 3, 3)
                a = a.view(5, 3)

                a = a / amass.acc_scale
                
                input = torch.cat([a.flatten(), ori.flatten()], dim=0).to("cuda")

                pose = net.forward_frame(input)

                poses.append(pose)
                
                pose = pose.cpu().numpy()      
                
                zero_tran = np.array([0, 0, 0])  
                viewer.update_all([pose], [zero_tran], render=False)
                viewer.render()
                
                idx += 1
                
                print('\r', clock.get_fps(), end='')
                
                if keyboard.is_pressed('q'):
                    break
            except Exception as e:
                print(f"Error occurred: {e}")
                print(traceback.format_exc())  # 打印完整的异常追踪信息
                os._exit(0)
            except KeyboardInterrupt:
                print("Exiting...")
                os._exit(0)
    
    print('\rFinish.')
        