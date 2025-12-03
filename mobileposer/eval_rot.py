import os
import torch
import articulate as art
import config
from model_tic import *
from TicOperator import *
from evaluation_functions import *


class TICModelEvaluator:
    def __init__(self, model_checkpoint_path, device='cuda:0', 
                 stack=3, 
                 n_input=config.imu_num * (3 + 3 * 3), 
                 n_output=config.imu_num * 6):
        # Set device and model
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = TIC(stack=stack, n_input=n_input, n_output=n_output)
        self.model.restore(model_checkpoint_path)

        self.model = self.model.to(self.device).eval()

        # set dataset path
        data_dir = "/root/autodl-tmp/processed_dataset/eval"
        dataset_name = 'imuposer_test.pt'
        self.dataset_path = os.path.join(data_dir, dataset_name)
        self.data = torch.load(self.dataset_path)
        
        # set body model
        self.body_model = art.ParametricModel(config.paths.smpl_file)
        
        # set combo
        self.combo = config.amass.combos['lw_rp_h']
        self.j_mask = torch.tensor([18, 19, 1, 2, 15, 0])

    def inference(self, acc, ori):
        acc = acc.to(self.device)
        ori = ori.to(self.device)

        ts = TicOperator(TIC_network=self.model, imu_num=config.imu_num, device=self.device)
        ori_fix, acc_fix, pred_drift, pred_offset = ts.run(ori, acc, trigger_t=1)

        return acc_fix.cpu(), ori_fix.cpu(), pred_drift.cpu(), pred_offset.cpu()
    
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

    def _print_results(self, result_static_ome, result_dynamic_ome=None):

        result_static_ome = torch.cat(result_static_ome, dim=0)
        result_dynamic_ome = torch.cat(result_dynamic_ome, dim=0) if result_dynamic_ome is not None else None


        print('=====Results=====')

        print('OME-static')
        print(result_static_ome.mean(dim=0), '|', result_static_ome.mean())

        print('OME-dynamic')
        if result_dynamic_ome is not None:
            print(result_dynamic_ome.mean(dim=0), '|', result_dynamic_ome.mean())

    def evaluate_imuposer(self, calibrate=False):
        result_static_ome = []
        result_dynamic_ome = []
        
        oris = self.data['ori'] # [N, imu_num, 3, 3]
        accs = self.data['acc'] # [N, imu_num, 3, 1]
        poses = self.data['pose']
        
        for idx in tqdm(range(len(oris))):
            ori, pose, acc = oris[idx], poses[idx], accs[idx]
            pose = self.body_model.forward_kinematics(pose, calc_mesh=False)[0].view(-1, 24, 3, 3)
            
            if calibrate:
                # calibrate imu data
                acc = acc[:, [0, 3, 4]]
                ori = ori[:, [0, 3, 4]]
                _, ori_calib, _, _ = self.inference(acc, ori)
                ori_calib = ori_calib.view(-1, config.imu_num, 3, 3)
            
            imu_bone = ori
            imu_bone_calib = ori_calib
            gt_bone  = pose[:, [18, 2, 15]]
            
            # calculate yaw
            gt_yaw = get_ego_yaw(gt_bone, ego_idx=-1)    # use head as ego id
            imu_yaw = get_ego_yaw(imu_bone, ego_idx=-1)  # use head as ego id
            imu_yaw_calib = get_ego_yaw(imu_bone_calib, ego_idx=-1)
            
            # transform rotation to ego-yaw
            gt_bone = gt_yaw.transpose(-2, -1).matmul(gt_bone)
            imu_bone = imu_yaw.transpose(-2, -1).matmul(imu_bone)
            imu_bone_calib = imu_yaw_calib.transpose(-2, -1).matmul(imu_bone_calib)
            
            # calculate errors
            err_static = angle_diff(imu_bone, gt_bone, imu_num=3, print_result=False)
            err_dynamic = angle_diff(imu_bone_calib, gt_bone, imu_num=3, print_result=False)
            
            result_static_ome.append(err_static)
            result_dynamic_ome.append(err_dynamic)
        self._print_results(result_static_ome, result_dynamic_ome)

def main():
    model_checkpoint_path = './checkpoint/TIC_20.pth'
    tic_evaluator = TICModelEvaluator(model_checkpoint_path)

    # Run evaluation
    tic_evaluator.evaluate_imuposer(calibrate=True)

if __name__ == "__main__":
    main()