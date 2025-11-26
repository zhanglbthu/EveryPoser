import socket
import threading
import numpy as np
import select
from scipy.spatial.transform import Rotation as R
import torch
import cv2
import time
import winsound
import articulate as art
import os
import traceback

KEYS = ['unix_timestamp', 'sensor_timestamp', 'accel_x', 'accel_y', 'accel_z', 'quart_x', 'quart_y', 'quart_z', 'quart_w', "roll", "pitch", "yaw"]

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
            try:
                readable, writable, exceptional = select.select(self.sockets, [], [])
                for sock in readable:
                    data, addr = sock.recvfrom(self.buffer_size)
                    # 调用 AppleSensor 的 process_data 方法处理接收到的数据
                    self.apple_sensor.process_data(data)
            except KeyboardInterrupt:
                print("Program interrupted by user.")
                self.stop()
                os._exit(0)
                break
    def stop(self):
        """停止接收线程"""
        self.running = False

# AppleSensor 类：实现数据接收、处理和获取最新数据
class AppleSensor:
    def __init__(self, udp_ports, device_ids, buffer_size=1024):
        self.sockets = []
        self.device_ids = device_ids
        for port in udp_ports:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("0.0.0.0", port))
            self.sockets.append(sock)

        # 初始化数据接收线程
        self.receiver = DataReceiver(self.sockets, buffer_size, self)
        self.receiver.start()

        # 初始化缓冲区
        self.raw_acc_buffer = {id: np.zeros((buffer_size, 3)) for id in device_ids.values()}
        self.raw_ori_buffer = {id: np.array([[0, 0, 0, 1]] * buffer_size) for id in device_ids.values()}
        self.timestamp_buffer = {id: np.zeros((buffer_size, 2)) for id in device_ids.values()}

    def get(self, device_id):
        """获取指定设备最新的加速度和四元数数据"""
        accel_data, quat_data, timestamp = self.get_latest_data(device_id)
        '''
        accel_data: [N, 3]
        quat_data: [N, 4] (xyzw)
        '''
        quat_data = quat_data[:, [3, 0, 1, 2]]  # 转换为 (wxyz)
        
        RIS = art.math.quaternion_to_rotation_matrix(torch.tensor(quat_data).float())  # [N, 3, 3]
        aS = - accel_data * 9.8
        aI = RIS.matmul(torch.tensor(aS, dtype=torch.float32).unsqueeze(-1)).squeeze(-1)
        
        return timestamp, aS, aI, RIS

    def get_latest_data(self, device_id):
        """获取指定设备的最新加速度和四元数数据"""
        # latest_acc = self.raw_acc_buffer[device_id][-1]
        # latest_ori = self.raw_ori_buffer[device_id][-1]
        latest_acc = [self.raw_acc_buffer[id][-1] for id in device_id]
        latest_ori = [self.raw_ori_buffer[id][-1] for id in device_id]
        latest_t = [self.timestamp_buffer[id][-1] for id in device_id]
        latest_acc = np.array(latest_acc)
        latest_ori = np.array(latest_ori)
        latest_t = np.array(latest_t)

        return latest_acc, latest_ori, latest_t

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
            os._exit(0)
            return

        data = []
        for d in data_str.strip().split(' '):
            try:
                data.append(float(d))
            except Exception as e:
                continue

        if len(data) != len(KEYS):
            if len(data) != len(KEYS) - 3:
                # something's missing, skip!
                print(list(np.array(data[-3:])*180/3.14))  # 可能是弧度转换成角度
                return

        # 根据设备ID确定设备名称
        if device_id == "left":
            device_name = self.device_ids[f"Left_{device_type}"]
        elif device_id == "right":
            device_name = self.device_ids[f"Right_{device_type}"]

        send_str = f"w{data[8]}wa{data[5]}ab{data[6]}bc{data[7]}c"  # 数据字符串（你可以根据需要修改）

        # 更新加速度和四元数数据
        curr_acc = np.array(data[2:5]).reshape(1, 3)
        curr_ori = np.array(data[5:9]).reshape(1, 4)
        timestamps = data[:2]

        # if device_name == 2:  # 如果是耳机设备
        #     # 转换四元数为欧拉角并调整顺序
        #     curr_euler = R.from_quat(curr_ori).as_euler("xyz").squeeze()
        #     fixed_euler = np.array([[curr_euler[0] * -1, curr_euler[2], curr_euler[1]]])
        #     curr_ori = R.from_euler("xyz", fixed_euler).as_quat().reshape(1, 4)
        #     curr_acc = np.array([[curr_acc[0, 0] * -1, curr_acc[0, 2], curr_acc[0, 1]]])

        # 更新缓冲区
        self.raw_acc_buffer[device_name] = np.concatenate([self.raw_acc_buffer[device_name][1:], curr_acc])
        self.raw_ori_buffer[device_name] = np.concatenate([self.raw_ori_buffer[device_name][1:], curr_ori])
        self.timestamp_buffer[device_name] = np.concatenate([self.timestamp_buffer[device_name][1:], np.array([timestamps])])

# Calibrated AppleSensor 类：在 AppleSensor 基础上添加校准功能
class CalibratedAppleSensor(AppleSensor):
    # _RMB_Npose = torch.tensor([[[0, 1, 0], [-1, 0, 0], [0, 0, 1]],         # left wrist
    #                            [[0, -1, 0], [1, 0, 0], [0, 0, 1]],         # right wrist
    #                            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],          # left thigh
    #                            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],          # right thigh
    #                            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],          # head
    #                            [[1, 0, 0], [0, 1, 0], [0, 0, 1]]]).float() # pelvis
    
    _RMB_Npose = torch.tensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1]],            # right thigh
                               [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],           # left wrist
                               [[1, 0, 0], [0, 1, 0], [0, 0, 1]]]).float()   # head
    
    def __init__(self, udp_ports, device_ids, buffer_size=1024):
        super().__init__(udp_ports, device_ids, buffer_size)
        self.ids = [id for id in device_ids.values()]
        self.N = len(self.ids)
        self.RMI = torch.eye(3).repeat(self.N, 1, 1)
        self.RSB = torch.eye(3).repeat(self.N, 1, 1)
        self.mask = [0, 1, 2]

    def get(self):
        t, aS, aI, RIS = super().get(self.ids) # aS: [N, 3], RIS: [N, 3, 3] 

        RMB = self.RMI.matmul(RIS).matmul(self.RSB)
        aM = self.RMI.matmul(aI.unsqueeze(-1)).squeeze(-1)
        
        return t, aS, aI, aM, RIS, RMB

    @staticmethod
    def _normalize_tensor(x: torch.Tensor, dim=-1):
        norm = x.norm(dim=dim, keepdim=True)
        normalized_x = x / norm
        return normalized_x

    @staticmethod
    def _mean_rotation(R0, R1):
        R_avg = (R0 + R1) / 2
        U, S, V = torch.svd(R_avg)
        R = torch.matmul(U, V.transpose(-2, -1)).view(-1, 3, 3)
        m = R.det() < 0
        R[m] = U[m].matmul(torch.diag(torch.tensor([1, 1, -1.], device=R.device))).bmm(V[m].transpose(1, 2))
        return R
    @staticmethod
    def _rotation_matrix_to_axis_angle(r):
        result = [cv2.Rodrigues(_)[0] for _ in r.view(-1, 3, 3).numpy()]
        result = torch.from_numpy(np.stack(result)).float().squeeze(-1)
        return result
    @staticmethod
    def _input(s, skip_input):
        if skip_input:
            print(s)
        else:
            return input(s)

    def _walking_calibration(self, RMI_method, skip_input=False):
        self._input('Apple: Stand in N pose for 3 seconds, then step forward and stop in the N-pose again.', skip_input)
        time.sleep(3)
        RIS_N0 = self.get()[4]

        winsound.Beep(440, 600)
        print('Step forward now.')
        begin_t = last_t = self.get()[0][0][1]
        p, v = torch.zeros(self.N, 3), torch.zeros(self.N, 3)
        Cov_pv = Cov_vv = 0
        while last_t - begin_t < 3:
            t, _, aI, _, _, _ = self.get()
            t = t[0][1]
            if t != last_t:
                dt = t - last_t
                last_t = t
                p += dt * v + 0.5 * dt * dt * aI
                v += dt * aI
                Cov_pv += dt * Cov_vv
                Cov_vv += 1
        p_filtered = p - Cov_pv / Cov_vv * v
        RIS_N1 = self.get()[4]

        if RMI_method == '9dof':
            zI = p_filtered.mean(dim=0).repeat(self.N, 1)
        elif RMI_method == '6dof':
            zI = p_filtered
        yI = torch.tensor([0, 0, 1.]).expand(self.N, 3)
        xI = self._normalize_tensor(yI.cross(zI, dim=-1))
        zI = self._normalize_tensor(xI.cross(yI, dim=-1))
        RMI = torch.stack([xI, yI, zI], dim=-2)
        
        RSB0 = RMI.matmul(RIS_N0).transpose(1, 2).matmul(self._RMB_Npose)[self.mask]
        RSB1 = RMI.matmul(RIS_N1).transpose(1, 2).matmul(self._RMB_Npose)[self.mask]
        
        RSB = self._mean_rotation(RSB0, RSB1)

        # import matplotlib.pyplot as plt
        # plt.scatter([0], [0], label='origin')
        # plt.scatter(p[:, 0], p[:, 1], label='raw')
        # plt.scatter(p_filtered[:, 0], p_filtered[:, 1], label='filtered')
        # plt.legend()
        # plt.show()

        err_vertical = p_filtered[:, -1].abs()
        err_RSB = self._rotation_matrix_to_axis_angle(RSB0.bmm(RSB1.transpose(1, 2))).norm(dim=-1) * (180 / np.pi)
        # change err_vertical from 0.1 to 0.4
        if all(err_vertical < 0.4) and all(err_RSB < 20) or skip_input:
            c = 'n'
            print('Calibration succeed: vertical error %s m, RSB error %s deg' % (err_vertical, err_RSB))
        else:
            c = input('Calibration fail: vertical error %s m, RSB error %s deg. Try again? [y]/n' % (err_vertical, err_RSB))
        if c != 'n':
            self._walking_calibration(RMI_method, skip_input)
        else:
            self.RMI = RMI
            self.RSB = RSB

    def calibrate(self, method, skip_input=False):
        r"""
        Calibrate the IMU set.

        :param method: Calibration method. Select from:
            - 'tpose_9dof':     Full T-pose calibration for 9-dof IMU. Two steps: align imu 1 and stand in T-pose.
            - 'tpose_6dof':     Full T-pose calibration for 6-dof IMU. Two steps: align all imus and stand in T-pose.
            - 'tpose_onlyRSB':  T-pose calibration only for sensor-to-bone offset. One step: stand in T-pose.
            - 'npose_9dof':     Full N-pose calibration for 9-dof IMU. Two steps: align imu 1 and stand in N-pose.
            - 'npose_6dof':     Full N-pose calibration for 6-dof IMU. Two steps: align all imus and stand in N-pose.
            - 'npose_onlyRSB':  N-pose calibration only for sensor-to-bone offset. One step: stand in N-pose.
            - 'npose_tpose':    Change-pose calibration for 9-dof IMU. One step: stand in N pose and then change to T-pose.
            - 'walking_9dof':   Walking calibration for 9-dof IMU. One step: stand in N pose, step forward and stop in N pose.
            - 'walking_6dof':   Walking calibration for 6-dof IMU. One step: stand in N pose, step forward and stop in N pose.
        :param skip_input: If true, skip user input and do calibration directly without error check.
        """
        if method == 'tpose_9dof':
            self._fixpose_calibration(RMI_method='9dof', RSB_method='tpose', skip_input=skip_input)
        elif method == 'tpose_6dof':
            self._fixpose_calibration(RMI_method='6dof', RSB_method='tpose', skip_input=skip_input)
        elif method == 'tpose_onlyRSB':
            self._fixpose_calibration(RMI_method='skip', RSB_method='tpose', skip_input=skip_input)
        elif method == 'npose_9dof':
            self._fixpose_calibration(RMI_method='9dof', RSB_method='npose', skip_input=skip_input)
        elif method == 'npose_6dof':
            self._fixpose_calibration(RMI_method='6dof', RSB_method='npose', skip_input=skip_input)
        elif method == 'npose_onlyRSB':
            self._fixpose_calibration(RMI_method='skip', RSB_method='npose', skip_input=skip_input)
        elif method == 'npose_tpose':
            self._changepose_calibration(skip_input=skip_input)
        elif method == 'walking_9dof':
            self._walking_calibration(RMI_method='9dof', skip_input=skip_input)
        elif method == 'walking_6dof':
            self._walking_calibration(RMI_method='6dof', skip_input=skip_input)