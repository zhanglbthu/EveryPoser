import threading
import socket
import numpy as np
import time
import select
from pygame.time import Clock
from articulate.utils.bullet.view_rotation_np import RotationViewer
from scipy.spatial.transform import Rotation as R

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
                try:
                    data, addr = sock.recvfrom(self.buffer_size)
                    # 调用 AppleSensor 的 process_data 方法处理接收到的数据
                    self.apple_sensor.process_data(data)

                except Exception as e:
                    print(e)

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

# 主程序代码
if __name__ == '__main__':
    udp_ports = [8001, 8002, 8003, 8004, 8005]
    device_ids = {
        "Left_phone": 0,
        "Left_watch": 1,
        "Left_headphone": 2,
        "Right_phone": 3,
        "Right_watch": 4
    }

    apple_sensor = AppleSensor(udp_ports, device_ids)
    clock = Clock()

    rviewer = RotationViewer(3, order='wxyz')  # RotationViewer 用于可视化
    rviewer.connect()

    device_ids_vis = [0, 1, 2]
    while True:
        try:
            clock.tick(25)
            r_list = []

            # 获取指定设备的最新数据
            for device_id in device_ids_vis:
                accel_data, quat_data = apple_sensor.get(device_id)

                # 将四元数添加到列表中
                r_list.append(quat_data)

            # 更新可视化
            rviewer.update_all(r_list)

        except KeyboardInterrupt:
            print("KeyboardInterrupt detected. Exiting the program.")
            AppleSensor.receiver.stop()
            break  # 退出循环
        except Exception as e:
            print(e)
            AppleSensor.receiver.stop()
            break  # 退出循环