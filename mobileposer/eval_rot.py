import os
import torch
import articulate as art
import config
from model_tic import *
# from TicOperator import *
from TicOperator_ours import *
from evaluation_functions import *
import matplotlib.pyplot as plt

class TICModelEvaluator:
    def __init__(self, model_checkpoint_path, device='cuda:0', 
                 stack=3, 
                 n_input=config.imu_num * (3 + 3 * 3), 
                 n_output=config.imu_num * 6):
        # Set device and model
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        # self.model = TIC(stack=stack, n_input=n_input, n_output=n_output)
        self.model = LSTMIC(n_input=n_input, n_output=n_output)
        self.model.restore(model_checkpoint_path)

        self.model = self.model.to(self.device).eval()

        # set dataset path
        data_dir = "/root/autodl-tmp/processed_dataset/eval"
        dataset_name = 'imuposer_full_upper_body.pt'
        self.dataset_path = os.path.join(data_dir, dataset_name)
        self.data = torch.load(self.dataset_path)
        
        # set body model
        self.body_model = art.ParametricModel(config.paths.smpl_file)
        
        # set ego id
        self.ego_id = -1
        
        # print ego id
        print(f'Ego ID: {self.ego_id}')

    def inference(self, acc, ori):
        acc = acc.to(self.device)
        ori = ori.to(self.device)

        ts = TicOperator(TIC_network=self.model, imu_num=config.imu_num, device=self.device)
        ori_fix, acc_fix, pred_drift, pred_offset = ts.run(ori, acc, trigger_t=1)

        return acc_fix.cpu(), ori_fix.cpu(), pred_drift.cpu(), pred_offset.cpu()

    def inference_ours(self, acc, ori):
        acc = acc.to(self.device)
        ori = ori.to(self.device)

        ts = TicOperator(TIC_network=self.model, imu_num=config.imu_num, data_frame_rate=30)
        ori_fix, acc_fix, pred_drift, pred_offset = ts.run_per_frame(ori, acc)

        return acc_fix.cpu(), ori_fix.cpu(), pred_drift, pred_offset
    
    def evaluate(self):
        print('=====Evaluation Start=====')

        result_static_ome = []
        result_dynamic_ome = []

        for f in self.folders:
            print(f'processing {f}')
            data_root = os.path.join(config.paths.tic_dataset_dir, f)

            # Load and process ground truth pose data
            gt_pose = torch.load(os.path.join(data_root, 'pose.pt'))
            gt_pose = axis_angle_to_rotation_matrix(gt_pose.reshape(-1, 3)).reshape(-1, 24, 3, 3)
            body_model = art.ParametricModel('./SMPL_MALE.pkl')
            body_shape = torch.zeros(10)
            trans = torch.zeros(3)
            grot, joint = body_model.forward_kinematics(gt_pose, body_shape, trans, calc_mesh=False)
            gt_bone = grot[:, [18, 19, 4, 5, 15, 0]]
            imu_bone = torch.load(os.path.join(data_root, 'rot.pt'))
            tic_fix_bone = torch.load(os.path.join(data_root, f'rot_fix_{self.tag}.pt')).reshape(-1, 6, 3, 3)

            # Acceleration data
            gt_acc = torch.load(os.path.join(data_root, 'acc_gt.pt')).reshape(-1, 6, 3, 1)
            imu_acc = torch.load(os.path.join(data_root, 'acc.pt')).reshape(-1, 6, 3, 1)
            tic_fix_acc = torch.load(os.path.join(data_root, f'acc_fix_{self.tag}.pt')).reshape(-1, 6, 3, 1)

            # Calculate yaw
            gt_yaw = get_ego_yaw(gt_bone)
            imu_yaw = get_ego_yaw(imu_bone)
            tic_fix_yaw = get_ego_yaw(tic_fix_bone)

            # Transform rotation to ego-yaw
            gt_bone = gt_yaw.transpose(-2, -1).matmul(gt_bone)
            imu_bone = imu_yaw.transpose(-2, -1).matmul(imu_bone)
            tic_fix_bone = tic_fix_yaw.transpose(-2, -1).matmul(tic_fix_bone)

            # Transform acceleration to ego-yaw
            gt_acc = gt_yaw.transpose(-2, -1).matmul(gt_acc)
            imu_acc = imu_yaw.transpose(-2, -1).matmul(imu_acc)
            tic_fix_acc = tic_fix_yaw.transpose(-2, -1).matmul(tic_fix_acc)

            # Compute errors
            err_static = angle_diff(imu_bone, gt_bone)
            result_static_ome.append(err_static)

            err_dynamic = angle_diff(tic_fix_bone, gt_bone)
            result_dynamic_ome.append(err_dynamic)

    def _print_results(self, result_static_ome, result_dynamic_ome=None, result_static_ome_global=None):

        result_static_ome = torch.cat(result_static_ome, dim=0)
        
        print('=====Results=====')

        print('OME-static')
        print(result_static_ome.mean(dim=0), '|', result_static_ome.mean())

        if result_dynamic_ome is not None:
            print('OME-dynamic')
            result_dynamic_ome = torch.cat(result_dynamic_ome, dim=0)
            print(result_dynamic_ome.mean(dim=0), '|', result_dynamic_ome.mean())
        if result_static_ome_global is not None:
            result_static_ome_global = torch.cat(result_static_ome_global, dim=0)
            print('OME-static-global')
            print(result_static_ome_global.mean(dim=0), '|', result_static_ome_global.mean())

    def evaluate_imuposer(self, calibrate: bool = False, save_dir: str = None, save_img: bool = False):
        result_static_ome = []
        result_static_ome_global = []
        result_dynamic_ome = []  # 只有 calibrate=True 时才会用到

        oris = self.data["ori"]   # [N, imu_num, 3, 3]
        accs = self.data["acc"]   # [N, imu_num, 3, 1]
        poses = self.data["pose"]

        for idx in tqdm(range(len(oris))):
            ori = oris[idx]
            acc = accs[idx]
            pose = poses[idx]

            pose = self.body_model.forward_kinematics(pose, calc_mesh=False)[0].view(-1, 24, 3, 3)
            gt_bone = pose[:, [18, 2, 15]]  # [T, 3, 3, 3] 之类（按你原本逻辑）
            ori = ori[:, [0, 3, 4]]  
            
            global_err = angle_diff(ori, gt_bone, imu_num=3, print_result=False)
            result_static_ome_global.append(global_err)
            
            # TODO: 基于ori和gt_bone可视化旋转误差欧拉角形式的三个分量并保存图片
            if save_img:
                R_err = ori.transpose(-2, -1).matmul(gt_bone)  
                
                euler_err = art.math.rotation_matrix_to_euler_angle(R_err, seq='YZX') * 180 / np.pi
                euler_err = euler_err.view(-1, 3, 3)
                seq_dir = os.path.join(save_dir, f'seq_{idx}')
                os.makedirs(seq_dir, exist_ok=True)
                
                T = euler_err.shape[0]
                x = torch.arange(T).cpu().numpy()
                
                for sensor_id in range(euler_err.shape[1]):
                    y = euler_err[:, sensor_id, :].detach().cpu().numpy()  # [T, 3]
                    
                    plt.figure()
                    plt.plot(x, y[:, 0], label="yaw (Y)")
                    plt.plot(x, y[:, 1], label="roll (Z)")
                    plt.plot(x, y[:, 2], label="pitch (X)")
                    
                    plt.ylim(-45, 45)
                    
                    plt.xlabel("frame")
                    plt.ylabel("euler error (deg)")
                    plt.title(f"Seq {idx:04d} | Sensor {sensor_id} | Euler Error")
                    plt.legend()
                    plt.tight_layout()
                    
                    out_path = os.path.join(seq_dir, f"imu_{sensor_id:02d}.png")
                    plt.savefig(out_path, dpi=200)
                    plt.close()
                    
            # 统一准备需要评估的 imu streams：至少 static；如果 calibrate 再加一个 calib
            imu_streams = [("static", ori)]
            if calibrate:
                acc_sel = acc[:, [0, 3, 4]]
                ori_sel = ori[:, [0, 3, 4]]
                _, ori_calib, _, _ = self.inference_ours(acc_sel, ori_sel)
                ori_calib = ori_calib.view(-1, config.imu_num, 3, 3)

                imu_streams.append(("calib", ori_calib))

            # 只算一次 gt_yaw
            gt_yaw = get_ego_yaw(gt_bone, ego_idx=self.ego_id)
            gt_bone_ego = gt_yaw.transpose(-2, -1).matmul(gt_bone)

            # 对每个 stream 统一做 yaw 对齐 + 误差计算
            for tag, imu_bone in imu_streams:
                imu_yaw = get_ego_yaw(imu_bone, ego_idx=self.ego_id)
                imu_bone_ego = imu_yaw.transpose(-2, -1).matmul(imu_bone)

                err = angle_diff(imu_bone_ego, gt_bone_ego, imu_num=3, print_result=False)

                if tag == "static":
                    result_static_ome.append(err)
                else:  # "calib"
                    result_dynamic_ome.append(err)

        # 输出也收敛一下
        if calibrate:
            self._print_results(result_static_ome, result_dynamic_ome)
        else:
            self._print_results(result_static_ome, result_static_ome_global=result_static_ome_global)

def main():
    # 构建模型文件路径
    model_checkpoint_path = f'./data/checkpoint/calibrator/Ours_IMUPoserData/Ours_IMUPoserData_20.pth'
        
    # 创建TIC模型评估器
    tic_evaluator = TICModelEvaluator(model_checkpoint_path)

    # 运行评估
    print(f'Evaluating TIC model with checkpoint: {model_checkpoint_path}')
    tic_evaluator.evaluate_imuposer(calibrate=False, save_dir='data/rotation/upper_body', save_img=True)

if __name__ == '__main__':
    main()