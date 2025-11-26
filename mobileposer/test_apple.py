from pygame.time import Clock
from articulate.utils.bullet.view_rotation_np import RotationViewer
from articulate.utils.pygame import StreamingDataViewer
from scipy.spatial.transform import Rotation as R
from argparse import ArgumentParser
from apple_sensor.sensor import AppleSensor, CalibratedAppleSensor
from config import AppleDevices
from articulate.utils.noitom.PN_lab import IMUSet, CalibratedIMUSet
import articulate as art
import torch
import os
import traceback
import numpy as np

np.set_printoptions(precision=10, suppress=True)
# 主程序代码
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--apple', action='store_true', help='use apple imu')
    parser.add_argument('--noitom', action='store_true', help='use noitom imu')
    parser.add_argument('--view_ori', action='store_true', help='view orientation data')
    parser.add_argument('--view_acc', action='store_true', help='view acceleration data')
    parser.add_argument('--view_t', action='store_true', help='view timestamp data')
    args = parser.parse_args()

    clock = Clock()

    if args.view_acc:
        sviewer = StreamingDataViewer(3, y_range=(-10, 10), window_length=200, names=['X', 'Y', 'Z']); sviewer.connect()
    if args.view_ori:
        rviewer = RotationViewer(2, order='wxyz'); rviewer.connect()
        rviewer_quan = StreamingDataViewer(3, y_range=(-90, 90), window_length=200, names=['x', 'y', 'z']); rviewer_quan.connect()
    if args.noitom:
        imu_set = CalibratedIMUSet()
        imu_set.calibrate('walking_9dof')
    if args.apple:
        apple_sensor = CalibratedAppleSensor(AppleDevices.udp_ports, AppleDevices.device_ids)
        apple_sensor.calibrate("walking_6dof")
    ids = [id for id in AppleDevices.device_ids.values()]
    print(f"Listening to Apple devices with IDs: {ids}")
    last_t = None
    while True:
        try:
            clock.tick(22)
            r_list = []

            if args.apple and not args.noitom:
                t, aS, aI, aM, RIS, RMB = apple_sensor.get()
            elif args.noitom and not args.apple:
                _, RIS, aS, wS, mS, aI, wI, mI, RMB, aM, wM, mM = imu_set.get()
            elif args.apple and args.noitom:
                RMB_apple = apple_sensor.get()[-1]
                RMB_gt = imu_set.get()[-4]
            if args.view_ori:
                # qulitative visualization
                R_apple = RMB_apple[2]
                R_gt = RMB_gt[4]
                q_apple = R.from_matrix(R_apple).as_quat()
                q_gt = R.from_matrix(R_gt).as_quat()
                # convert to wxyz
                q_apple = np.array([q_apple[3], q_apple[0], q_apple[1], q_apple[2]])
                q_gt = np.array([q_gt[3], q_gt[0], q_gt[1], q_gt[2]])
                rviewer.update_all([q_gt, q_apple])
                
                # quantitative visualization
                delta_R = R_apple.T @ R_gt
                delta_euler = R.from_matrix(delta_R).as_euler('xyz', degrees=True)
                rviewer_quan.plot(delta_euler)
                
            if args.view_acc:
                # process acceleration data
                sviewer.plot(aI[2])
            if args.view_t:
                t = t[0][1] # select the first device's timestamp
                if last_t is None:
                    last_t = t
                else:
                    dt = t - last_t
                    last_t = t
                    print(f"dt: {dt*1000:.5f} ms")
        except Exception as e:
            print(f"Error occurred: {e}")
            print(traceback.format_exc())  # 打印完整的异常追踪信息
            os._exit(0)
        except KeyboardInterrupt:
            print("Exiting...")
            os._exit(0)
