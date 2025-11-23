import time
import torch
from pygame.time import Clock

import articulate as art
import os
from config import *
from articulate.utils.noitom import *
from articulate.utils.unity import MotionViewer

from articulate.utils.wearable import WearableSensorSet
from articulate.utils.pygame import StreamingDataViewer
from articulate.utils.bullet.view_rotation_np import RotationViewer
from auxiliary import calibrate_q, quaternion_inverse
import select
from utils.model_utils import load_model
import numpy as np
import matplotlib
from argparse import ArgumentParser
import keyboard
import threading
import socket

from scipy.spatial.transform import Rotation as R

colors = matplotlib.colormaps['tab10'].colors
body_model = art.ParametricModel(paths.smpl_file, device='cuda')

# 定义常量和数据结构
BUFFER_SIZE = 50  # 数据缓冲区大小
KEYS = ['unix_timestamp', 'sensor_timestamp', 'accel_x', 'accel_y', 'accel_z', 'quart_x', 'quart_y', 'quart_z', 'quart_w', "roll", "pitch", "yaw"]
device_ids = {
    "Left_phone": 0,
    "Left_watch": 1,
    "Left_headphone": 2,
    "Right_phone": 3,
    "Right_watch": 4
}
import keyboard

# 用于接收数据的线程类
class DataReceiver(threading.Thread):
    def __init__(self, sockets, buffer_size=1024, apple_sensor=None):
        super().__init__()
        self.sockets = sockets
        self.buffer_size = buffer_size
        self.apple_sensor = apple_sensor  # 引用 AppleSensor 实例
        self.running = True

    def run(self):
        """持续接收数据并更新缓冲区"""
        while self.running:
            readable, writable, exceptional = select.select(self.sockets, [], [])
            for sock in readable:
                data, addr = sock.recvfrom(self.buffer_size)
                # 调用 AppleSensor 的 process_data 方法处理接收到的数据
                self.apple_sensor.process_data(data)
                    

    def stop(self):
        """停止接收线程"""
        self.running = False

# AppleSensor 类：实现数据接收、处理和获取最新数据
class AppleSensor:
    def __init__(self, udp_ports, device_ids, buffer_size=1024):
        self.sockets = []
        for port in udp_ports:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("0.0.0.0", port))
            self.sockets.append(sock)

        # 初始化数据接收线程
        self.receiver = DataReceiver(self.sockets, buffer_size, self)
        self.receiver.start()

        # 初始化缓冲区
        self.raw_acc_buffer = {id: np.zeros((BUFFER_SIZE, 3)) for id in device_ids.values()}
        self.raw_ori_buffer = {id: np.array([[0, 0, 0, 1]] * BUFFER_SIZE) for id in device_ids.values()}

    def get(self, device_id):
        """获取指定设备最新的加速度和四元数数据"""
        accel_data, quat_data = self.get_latest_data(device_id)
        # 转换四元数顺序 (wxyz)
        quat_data = np.array([quat_data[3], quat_data[0], quat_data[1], quat_data[2]])
        return accel_data, quat_data

    def get_latest_data(self, device_id):
        """获取指定设备的最新加速度和四元数数据"""
        latest_acc = self.raw_acc_buffer[device_id][-1]
        latest_ori = self.raw_ori_buffer[device_id][-1]
        return latest_acc, latest_ori

    def process_data(self, message):
        """Receive data from socket and process it."""
        message = message.strip()
        if not message:
            return
        message = message.decode('utf-8')
        if message == 'stop':
            return
        if ':' not in message:
            print(message)
            return

        try:
            device_id, raw_data_str = message.split(";")
            device_type, data_str = raw_data_str.split(':')
        except Exception as e:
            print(e, message)
            return

        data = []
        for d in data_str.strip().split(' '):
            try:
                data.append(float(d))
            except Exception as e:
                print(e)
                continue

        if len(data) != len(KEYS):
            if len(data) != len(KEYS) - 3:
                # something's missing, skip!
                print(list(np.array(data[-3:])*180/3.14))  # 可能是弧度转换成角度
                return

        # 根据设备ID确定设备名称
        if device_id == "left":
            device_name = device_ids[f"Left_{device_type}"]
        elif device_id == "right":
            device_name = device_ids[f"Right_{device_type}"]

        send_str = f"w{data[8]}wa{data[5]}ab{data[6]}bc{data[7]}c"  # 数据字符串（你可以根据需要修改）

        # 更新加速度和四元数数据
        curr_acc = np.array(data[2:5]).reshape(1, 3)
        curr_ori = np.array(data[5:9]).reshape(1, 4)
        timestamps = data[:2]

        if device_name == 2:  # 如果是耳机设备
            # 转换四元数为欧拉角并调整顺序
            curr_euler = R.from_quat(curr_ori).as_euler("xyz").squeeze()
            fixed_euler = np.array([[curr_euler[0] * -1, curr_euler[2], curr_euler[1]]])
            curr_ori = R.from_euler("xyz", fixed_euler).as_quat().reshape(1, 4)
            curr_acc = np.array([[curr_acc[0, 0] * -1, curr_acc[0, 2], curr_acc[0, 1]]])

        # 更新缓冲区
        self.raw_acc_buffer[device_name] = np.concatenate([self.raw_acc_buffer[device_name][1:], curr_acc])
        self.raw_ori_buffer[device_name] = np.concatenate([self.raw_ori_buffer[device_name][1:], curr_ori])

class IMUSet:
    g = 9.8

    def __init__(self, udp_port=7777):
        app = MCPApplication()
        settings = MCPSettings()
        settings.set_udp(udp_port)
        settings.set_calc_data()
        app.set_settings(settings)
        app.open()
        time.sleep(0.5)

        sensors = [None for _ in range(6)]
        evts = []
        print('Waiting for sensors...')
        while len(evts) == 0:
            evts = app.poll_next_event()
            for evt in evts:  
                assert evt.event_type == MCPEventType.SensorModulesUpdated
                sensor_module_handle = evt.event_data.sensor_module_data.sensor_module_handle
                sensor_module = MCPSensorModule(sensor_module_handle)
                sensors[sensor_module.get_id() - 1] = sensor_module

        print('find %d sensors' % len([_ for _ in sensors if _ is not None]))
        self.app = app
        self.sensors = sensors
        self.t = 0

    def get(self):
        evts = self.app.poll_next_event()
        if len(evts) > 0:
            self.t = evts[0].timestamp
        q, a = [], []
        for sensor in self.sensors:
            q.append(sensor.get_posture())
            a.append(sensor.get_accelerated_velocity())

        # assuming g is positive (= 9.8), we need to change left-handed system to right-handed by reversing axis x, y, z
        R = art.math.quaternion_to_rotation_matrix(torch.tensor(q))  # rotation is not changed
        a = -torch.tensor(a) / 1000 * self.g                         # acceleration is reversed
        a = R.bmm(a.unsqueeze(-1)).squeeze(-1) + torch.tensor([0., 0., self.g])   # calculate global free acceleration
        return self.t, R, a

    def clear(self):
        pass

def align_sensor(apple_sensor : AppleSensor, n_calibration : int):
    r"""Align noitom and sensor imu data"""
    print('Rotate the sensor & imu together.')
    qIC_list, qOS_list = [], []
    
    for i in range(n_calibration):
        qIS, qCO = [], []

        # qIS.append(torch.tensor([0., 0., 1.0, 0.]).float()) # noitom
        qIS_ref = imu_set.sensors[0].get_posture()
        qIS.append(torch.tensor(qIS_ref).float()) # noitom
        
        while len(qCO) < 1:
            _, q_sensor = apple_sensor.get(0)
            qCO.append(torch.tensor(q_sensor).float())
            print('\rCalibrating... (%d/%d)' % (i, n_calibration), end='')
            
        qCI, qSO = calibrate_q(torch.stack(qIS), torch.stack(qCO))
        print('\tfinished\nqCI:', qCI, '\tqSO:', qSO)
        qIC_list.append(quaternion_inverse(qCI))
        qOS_list.append(quaternion_inverse(qSO))
    return qIC_list, qOS_list

def tpose_calibration_noitom(imu_set):
     print('Calibrating T-pose...')
     c = input('Used cached RMI? [y]/n    (If you choose no, put imu 1 straight (x = Right, y = Forward, z = Down, Left-handed).')
     if c == 'n' or c == 'N':
         imu_set.clear()
         RSI_gt = imu_set.get()[1][0].view(3, 3).t()
         RMI_gt = torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0.]]).mm(RSI_gt)
         torch.save(RMI_gt, os.path.join(paths.temp_dir, 'RMI.pt'))
     else:
         RMI_gt = torch.load(os.path.join(paths.temp_dir, 'RMI.pt'))

     print(RMI_gt)
 
     input('Stand straight in T-pose and press enter. The calibration will begin in 3 seconds')
     time.sleep(3)
     imu_set.clear()
     RIS_gt = imu_set.get()[1]

     RSB_gt = RMI_gt.matmul(RIS_gt).transpose(1, 2).matmul(torch.eye(3))
     return RMI_gt, RSB_gt

def tpose_calibration_sensor(apple_sensor : AppleSensor, n_calibration : int):
    
    input('Stand straight in T-pose and press enter. The Apple calibration will begin in 3 seconds')
    time.sleep(3)
    
    RIS = torch.eye(3).repeat(6, 1, 1)
    
    for i in range(n_calibration):
        _, q_sensor = apple_sensor.get(i)
        qCO_sensor = torch.tensor(q_sensor).float()
        qIS_sensor = art.math.quaternion_product(art.math.quaternion_product(qIC_list[i], qCO_sensor), qOS_list[i])
        RIS_sensor = art.math.quaternion_to_rotation_matrix(qIS_sensor)
        
        if i == 0:
            index = 3
        elif i == 1:
            index = 0
        elif i == 2:
            index = 4
            
        RIS[index, :, :] = RIS_sensor[0, :, :]
    
    RSB = RMI.matmul(RIS).transpose(1, 2).matmul(torch.eye(3))
    return RSB

def align_sensor_noitom(imu_set, sensor_set, n_calibration):
     r"""Align noitom and sensor imu data"""
     print('Rotate the sensor & imu together.')
     qIC_list, qOS_list = [], []
 
     for i in range(n_calibration):
         qIS, qCO = [], []
         while len(qIS) < 1:
             imu_set.app.poll_next_event()
             sensor_data = sensor_set.get()
             if not 0 in sensor_data:
                 continue
 
             qIS.append(torch.tensor(imu_set.sensors[0].get_posture()).float()) # noitom
             qCO.append(torch.tensor(sensor_data[i].orientation).float()) # wearable sensor
             print('\rCalibrating... (%d/%d)' % (i, n_calibration), end='')
 
         qCI, qSO = calibrate_q(torch.stack(qIS), torch.stack(qCO))
         print('\tfinished\nqCI:', qCI, '\tqSO:', qSO)
         qIC_list.append(quaternion_inverse(qCI))
         qOS_list.append(quaternion_inverse(qSO))
     return qIC_list, qOS_list

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
        imu_set = IMUSet()
        RMI, RSB = tpose_calibration_noitom(imu_set=imu_set)
    
    if args.apple:
        udp_ports = [8001, 8002, 8003, 8004, 8005]
        device_ids = {
            "Left_phone": 0,
            "Left_watch": 1,
            "Left_headphone": 2,
            "Right_phone": 3,
            "Right_watch": 4
        }

        apple_sensor = AppleSensor(udp_ports, device_ids)
        n_calibration = 3
        qIC_list, qOS_list = align_sensor(apple_sensor, n_calibration)
        RSB_sensor = tpose_calibration_sensor(apple_sensor, n_calibration)
    
    accs, oris, poses, trans = [], [], [], []
    
    idx = 0
    # sviewer = StreamingDataViewer(3, y_range=(-10, 10), window_length=200, names=['X', 'Y', 'Z']); sviewer.connect()
    # rviewer = RotationViewer(2, order='wxyz'); rviewer.connect()
    with torch.no_grad(), MotionViewer(1, overlap=False, names=['mocap']) as viewer:
        while True:
            clock.tick(30)
            viewer.clear_all(render=False)
            
            if args.noitom:
                # gt readings
                tframe, RIS, aI = imu_set.get()
                
                RMB = torch.zeros_like(RIS).to(device)
                aM = torch.zeros_like(aI).to(device)
                
                RMB[combo] = RMI.matmul(RIS).matmul(RSB)[combo].to(device)
                aM[combo] = aI.mm(RMI.t())[combo].to(device)

                # [6, 3, 3] -> [5, 3, 3], 去掉最后一个pelvis
                RMB = RMB[[0, 1, 2, 3, 4], :, :]
                aM = aM[[0, 1, 2, 3, 4], :]
            
            if args.apple:
                RIS_ours = torch.eye(3).repeat(6, 1, 1)
                aI_ours = torch.zeros(6, 3)
                for i in range(n_calibration):
                    acc, rot_q = apple_sensor.get(i)
                    qCO_sensor = torch.tensor(rot_q).float()
                    aSS_sensor = torch.tensor(acc).float()
                    qIS_sensor = art.math.quaternion_product(art.math.quaternion_product(qIC_list[i], qCO_sensor), qOS_list[i])
                    RIS_sensor = art.math.quaternion_to_rotation_matrix(qIS_sensor)
                    
                    aIS_sensor = RIS_sensor.squeeze(0).mm( - aSS_sensor.unsqueeze(-1)).squeeze(-1) + torch.tensor([0., 0., 9.8])
                    
                    if i == 0:
                        index = 3
                        # sviewer.plot(aSS_sensor)
                    elif i == 1:
                        index = 0
                    elif i == 2:
                        index = 4
                        
                    else:
                        print("Unknown sensor index")
                        
                    RIS_ours[index, :, :] = RIS_sensor[0, :, :]
                    aI_ours[index, :] = aIS_sensor
                
                RMB_sensor = torch.zeros_like(RIS).to(device)
                aM_sensor = torch.zeros_like(aI).to(device)
                
                RMB_sensor[combo] = RMI.matmul(RIS_ours).matmul(RSB_sensor)[combo].to(device)
                aM_sensor[combo] = aI_ours.mm(RMI.t())[combo].to(device)
            
            # r_list = []
            # aa_gt = art.math.rotation_matrix_to_axis_angle(RMB[4]).view(3)
            # q_gt = art.math.axis_angle_to_quaternion(aa_gt).view(4)
            
            # r_list.append(q_gt.cpu().numpy())
            # aa_ours = art.math.rotation_matrix_to_axis_angle(RMB_sensor[4]).view(3)
            # q_ours = art.math.axis_angle_to_quaternion(aa_ours).view(4)
            # r_list.append(q_ours.cpu().numpy())
            
            # rviewer.update_all(r_list)
            
            # RMB_sensor = RMB_sensor.view(5, 3, 3)
            # aM_sensor = aM_sensor.view(5, 3)

            oris.append(RMB)
            accs.append(aM)

            RMB = RMB.view(5, 3, 3)
            aM = aM.view(5, 3)

            aM = aM / amass.acc_scale
            
            input = torch.cat([aM.flatten(), RMB.flatten()], dim=0).to("cuda")

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
    
    print('\rFinish.')
        