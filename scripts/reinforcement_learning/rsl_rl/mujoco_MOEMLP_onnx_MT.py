import argparse, os, time, sys
import numpy as np
import mujoco, mujoco_viewer
from tqdm import tqdm
from collections import deque
import onnxruntime as ort
import torch

network_script_directory = "/home/ubuntu/Desktop/robot_lab/source/robot_lab/robot_lab/tasks/manager_based/MotionTracking/config/g1/agents"
if network_script_directory not in sys.path:
    sys.path.append(network_script_directory)

# MOEMLP 不需要导入 Transformer 相关类，但保留导入避免其他依赖报错
from Transformer_ActorCritic import MOEMLPActorCritic
from tensordict import TensorDict
import utils.math_utils as math_utils

class HumanoidEnv:
    def __init__(self, policy_path, motion_path, robot_type="g1", device="cuda", record_video=False):
        self.robot_type = robot_type
        self.device = device
        self.record_video = record_video
        self.motion_path = motion_path
        self.policy_path = policy_path

        self.load_motion()
        self.load_model()  # ← 这里会设置 proprio_dim 等参数
        
        self.mujoco2isaac_dof_index = torch.tensor([0, 6, 12, 1, 7, 13, 18, 2, 8, 14, 19, 3, 9, 15, 20, 4, 10, 16, 21, 5, 11, 17, 22], device=self.device)
        self.isaac2mujoco_dof_index = torch.tensor([0, 3, 7, 11, 15, 19, 1, 4, 8, 12, 16, 20, 2, 5, 9, 13, 17, 21, 6, 10, 14, 18, 22], device=self.device)
        self.motion_body_indexes = torch.tensor([0, 5, 15, 23, 6, 16, 24, 4, 13, 21, 25, 14, 22, 26], device=self.device)
        self.robot_body_indexes = torch.tensor([1, 3, 5, 7, 9, 11, 13, 14, 16, 18, 19, 21, 23, 24], device=self.device)
        self.anchor_index = 4
        
        if robot_type == "g1":
            model_path = "/home/ubuntu/Desktop/hjq/assets/g1_23dof_rev_1_0.xml"
            self.stiffness = np.array([ 
                40.179, 99.098, 40.179, 99.098, 28.501, 28.501,
                40.179, 99.098, 40.179, 99.098, 28.501, 28.501,
                40.179,
                14.25, 14.25, 14.25, 14.25, 14.25,
                14.25, 14.25, 14.25, 14.25, 14.25
            ])
            self.damping = np.array([ 
                2.557, 6.308, 2.557, 6.308, 1.814, 1.814,
                2.557, 6.308, 2.557, 6.308, 1.814, 1.814,
                2.557,
                0.907, 0.907, 0.907, 0.907, 0.907,
                0.907, 0.907, 0.907, 0.907, 0.907,
            ])
            self.torque_limits = np.array([ 
                88.0, 139.0, 88.0, 139.0, 35.0, 35.0,
                88.0, 139.0, 88.0, 139.0, 35.0, 35.0,
                88.0,
                25.0, 25.0, 25.0, 25.0, 25.0,
                25.0, 25.0, 25.0, 25.0, 25.0
            ])
            self.default_dof_pos = torch.tensor([
                -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
                -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
                0.0,
                0.2, 0.2, 0.0, 0.6, 0.0,
                0.2, -0.2, 0.0, 0.6, 0.0,
            ], device=self.device, dtype=torch.float32)
            self.action_scale = torch.tensor([
                0.547, 0.350, 0.547, 0.350, 0.307, 0.307,
                0.547, 0.350, 0.547, 0.350, 0.307, 0.307,
                0.547,
                0.438, 0.438, 0.438, 0.438, 0.438,
                0.438, 0.438, 0.438, 0.438, 0.438
            ], device=self.device, dtype=torch.float32)
            self.num_actions = 23
            self.num_dofs = 23
        else:
            raise ValueError(f"Robot type {robot_type} not supported!")
        
        self.sim_duration = 60.0
        self.sim_dt = 0.005
        self.sim_decimation = 4
        self.control_dt = self.sim_dt * self.sim_decimation
        
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = self.sim_dt
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_step(self.model, self.data)

        self.align_motion_to_robot()
        if self.record_video:
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data, 'offscreen')
        else:
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        self.viewer.cam.distance = 5.0
        
        self.last_action = torch.zeros(self.num_actions, device=self.device, dtype=torch.float32)

        # MOEMLP 不需要 future_steps 相关，但保留变量避免其他代码报错
        self.tar_obs_steps = torch.arange(1, self.future_steps + 1, device=self.device, dtype=torch.int)
        
        self.proprio_history_buffer = deque(maxlen=self.history_length)
        for _ in range(self.history_length):
            # ← 关键：使用 load_model 中设置的 proprio_dim
            self.proprio_history_buffer.append(torch.zeros(self.policy_proprio_dim, device=self.device, dtype=torch.float32))


    def load_motion(self):
        data = np.load(self.motion_path, allow_pickle=True)
        self.joint_pos = torch.tensor(data['joint_pos'], device=self.device, dtype=torch.float32)
        self.joint_vel = torch.tensor(data['joint_vel'], device=self.device, dtype=torch.float32)
        self.body_pos_w = torch.tensor(data['body_pos_w'], device=self.device, dtype=torch.float32)
        self.body_quat_w = torch.tensor(data['body_quat_w'], device=self.device, dtype=torch.float32)
        self.body_lin_vel_w = torch.tensor(data['body_lin_vel_w'], device=self.device, dtype=torch.float32)
        self.body_ang_vel_w = torch.tensor(data['body_ang_vel_w'], device=self.device, dtype=torch.float32)
        self.body_pos_r = torch.tensor(data['body_pos_r'], device=self.device, dtype=torch.float32)
        self.body_quat_r = torch.tensor(data['body_quat_r'], device=self.device, dtype=torch.float32)
        self.body_lin_vel_r = torch.tensor(data['body_lin_vel_r'], device=self.device, dtype=torch.float32)
        self.body_ang_vel_r = torch.tensor(data['body_ang_vel_r'], device=self.device, dtype=torch.float32)
        self.motion_len = self.joint_pos.shape[0]


    def load_model(self):
        print(f"Loading ONNX policy from {self.policy_path}")
        
        # ========== MOEMLP 配置参数 ==========
        self.future_steps = 20          # 保留但不使用，避免其他代码报错
        self.history_length = 1
        self.num_actions = 23
        # ⚠️ 关键：MOEMLP 的 proprio_dim = 253 (根据导出脚本配置)
        # 如果报错维度不匹配，请打印 proprio_obs.shape 确认实际维度
        self.policy_proprio_dim = 253   # ← 从 292 改为 253
        # =====================================
        
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == "cuda" else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(self.policy_path, providers=providers)
        
        # ⚠️ MOEMLP 只有一个输入节点: proprio_history
        # 不再需要 future_motion 输入
        self.input_name_proprio = self.session.get_inputs()[0].name
        print(f"✓ ONNX input: {self.input_name_proprio}")


    def align_motion_to_robot(self):
        robot_pos = torch.from_numpy(self.data.body('torso_link').xpos.astype(np.float32)).to(self.device).unsqueeze(0)
        robot_quat = torch.from_numpy(self.data.body('torso_link').xquat.astype(np.float32)).to(self.device).unsqueeze(0)

        motion_pos_start = self.body_pos_w[0, self.anchor_index].unsqueeze(0)
        motion_quat_start = self.body_quat_w[0, self.anchor_index].unsqueeze(0)

        robot_yaw_quat = math_utils.yaw_quat(robot_quat)
        motion_yaw_quat = math_utils.yaw_quat(motion_quat_start)
        yaw_offset_quat = math_utils.quat_mul(robot_yaw_quat, math_utils.quat_inv(motion_yaw_quat))
        yaw_offset_quat_exp = yaw_offset_quat.view(1, 1, 4)
        pos_offset = robot_pos.clone()
        pos_offset[..., 2] = 0
        num_frames = self.body_pos_w.shape[0]
        num_bodies = self.body_pos_w.shape[1]
        m_pos_start_xy = motion_pos_start.clone()
        m_pos_start_xy[..., 2] = 0
        
        centered_pos = self.body_pos_w - m_pos_start_xy.view(1, 1, 3)
        self.body_pos_w = math_utils.quat_apply(
            yaw_offset_quat_exp.expand(num_frames, num_bodies, 4), 
            centered_pos.reshape(-1, 3)
        ).view(num_frames, num_bodies, 3) + pos_offset.view(1, 1, 3)

        self.body_lin_vel_w = math_utils.quat_apply(
            yaw_offset_quat_exp.expand(num_frames, num_bodies, 4),
            self.body_lin_vel_w.reshape(-1, 3)
        ).view(num_frames, num_bodies, 3)
        
        self.body_ang_vel_w = math_utils.quat_apply(
            yaw_offset_quat_exp.expand(num_frames, num_bodies, 4),
            self.body_ang_vel_w.reshape(-1, 3)
        ).view(num_frames, num_bodies, 3)

        self.body_quat_w = math_utils.quat_mul(
            yaw_offset_quat_exp.expand(num_frames, num_bodies, 4).reshape(-1, 4),
            self.body_quat_w.reshape(-1, 4)
        ).view(num_frames, num_bodies, 4)


    def get_future_obs(self, curr_timestep):
        """保留函数但不使用，避免删除导致其他依赖报错"""
        target_indices = self.tar_obs_steps + curr_timestep
        target_indices = torch.clamp(target_indices, max=self.motion_len - 1)
        # ... 原有逻辑保持不变，但 run() 中不再调用


    def get_proprio_obs(self, curr_timestep):
        dof_pos = torch.from_numpy(self.data.qpos[-self.num_dofs:].astype(np.float32)).to(self.device)
        dof_vel = torch.from_numpy(self.data.qvel[-self.num_dofs:].astype(np.float32)).to(self.device)
        anchor_pos_w = torch.from_numpy(self.data.body('torso_link').xpos.astype(np.float32)).to(self.device)
        anchor_quat_w = torch.from_numpy(self.data.body('torso_link').xquat.astype(np.float32)).to(self.device)
        base_lin_vel = torch.from_numpy(self.data.qvel.astype(np.float32)[0:3]).to(self.device)
        base_ang_vel = torch.from_numpy(self.data.qvel.astype(np.float32)[3:6]).to(self.device)

        gravity_vec_w = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=torch.float32)
        projected_gravity_b = math_utils.quat_apply_inverse(anchor_quat_w, gravity_vec_w)
        
        target_idx = min(curr_timestep, self.motion_len - 1)
        motion_joint_pos = self.joint_pos[target_idx]
        motion_joint_vel = self.joint_vel[target_idx]
        command = torch.cat([motion_joint_pos, motion_joint_vel], dim=-1)
        rel_pos_w = self.body_pos_w[target_idx, self.anchor_index] - anchor_pos_w
        motion_anchor_pos_b = math_utils.quat_apply_inverse(anchor_quat_w, rel_pos_w)
        rel_quat_b = math_utils.quat_mul(math_utils.quat_conjugate(anchor_quat_w), self.body_quat_w[target_idx, self.anchor_index])
        motion_anchor_ori_b = math_utils.matrix_from_quat(rel_quat_b)[..., :2].reshape(-1)

        robot_body_pos_w = torch.from_numpy(self.data.xpos.astype(np.float32)).to(self.device)[self.robot_body_indexes]
        robot_body_quat_w = torch.from_numpy(self.data.xquat.astype(np.float32)).to(self.device)[self.robot_body_indexes]
        robot_body_lin_vel_w = torch.from_numpy(self.data.cvel.astype(np.float32)).to(self.device)[self.robot_body_indexes, 3:]
        robot_body_ang_vel_w = torch.from_numpy(self.data.cvel.astype(np.float32)).to(self.device)[self.robot_body_indexes, :3]
        diff_p = robot_body_pos_w - anchor_pos_w.view(1, 3)
        anchor_quat_inv = math_utils.quat_conjugate(anchor_quat_w).view(1, 4).expand(len(self.robot_body_indexes), 4)
        robot_body_pos_r = math_utils.quat_apply(anchor_quat_inv, diff_p).reshape(-1)
        diff_q = math_utils.quat_mul(anchor_quat_inv, robot_body_quat_w)
        robot_body_ori_r = math_utils.matrix_from_quat(diff_q)[..., :2].reshape(-1)
        robot_body_lin_vel_r = math_utils.quat_apply(anchor_quat_inv, robot_body_lin_vel_w).reshape(-1)
        robot_body_ang_vel_r = math_utils.quat_apply(anchor_quat_inv, robot_body_ang_vel_w).reshape(-1)
        
        dof_pos = (dof_pos - self.default_dof_pos)[self.mujoco2isaac_dof_index]
        dof_vel = (dof_vel - 0)[self.mujoco2isaac_dof_index]
        
        proprio_obs = torch.cat([
            command,                    # 46
            # motion_anchor_pos_b,      # 3 (注释掉)
            motion_anchor_ori_b,        # 6
            robot_body_pos_r,           # 42
            robot_body_ori_r,           # 84
            # robot_body_lin_vel_r,     # 42 (注释掉)
            # robot_body_ang_vel_r,     # 42 (注释掉)
            # projected_gravity_b,      # 3 (注释掉)
            base_lin_vel,               # 3
            base_ang_vel,               # 3
            dof_pos,                    # 23
            dof_vel,                    # 23
            self.last_action,           # 23
        ], dim=-1)
        # 调试用：打印实际维度，确认是否与 policy_proprio_dim 一致
        # print(f"[DEBUG] proprio_obs.shape: {proprio_obs.shape}")
        return proprio_obs


    def run(self):
        motion_name = os.path.basename(self.motion_path).split('.')[0]
        if self.record_video:
            import imageio
            video_name = f"{self.robot_type}_{''.join(os.path.basename(self.policy_path).split('.')[:-1])}_{motion_name}.mp4"
            path = "mujoco_videos/"
            if not os.path.exists(path):
                os.makedirs(path)
            video_name = os.path.join(path, video_name)
            mp4_writer = imageio.get_writer(video_name, fps=50)
        
        self.pd_target = self.default_dof_pos.cpu().numpy()

        for i in tqdm(range(int(self.sim_duration / self.sim_dt)), desc="Running simulation..."):
            curr_timestep = i // self.sim_decimation
            proprio_obs = self.get_proprio_obs(curr_timestep)
            
            if i % self.sim_decimation == 0:
                # ⚠️ MOEMLP: 不再需要 future_obs，只传入 proprio_history
                self.proprio_history_buffer.append(proprio_obs)
                
                # [B=1, T, proprio_dim]
                policy_proprio_history = torch.stack(list(self.proprio_history_buffer), dim=0).unsqueeze(0).cpu().numpy()
                
                # ⚠️ 关键：ONNX 输入只有 proprio_history
                inputs = {
                    self.input_name_proprio: policy_proprio_history,
                }
                
                ort_outputs = self.session.run(None, inputs)
                action = ort_outputs[0].squeeze()
                action = torch.from_numpy(action).to(self.device)

                self.last_action = action.clone()               

                self.pd_target = action * self.action_scale[self.mujoco2isaac_dof_index] + self.default_dof_pos[self.mujoco2isaac_dof_index]
                self.pd_target = self.pd_target[self.isaac2mujoco_dof_index].cpu().numpy()
                
                self.viewer.cam.lookat = self.data.qpos.astype(np.float32)[:3]
                if self.record_video:
                    img = self.viewer.read_pixels()
                    mp4_writer.append_data(img)
                else:
                    self.viewer.render()
            
            # PD 控制
            dof_pos = self.data.qpos[-self.num_dofs:]
            dof_vel = self.data.qvel[-self.num_dofs:]
            torque = (self.pd_target - dof_pos) * self.stiffness - dof_vel * self.damping
            torque = np.clip(torque, -self.torque_limits, self.torque_limits)
            self.data.ctrl = torque
            
            mujoco.mj_step(self.model, self.data)
        
        self.viewer.close()
        if self.record_video:
            mp4_writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot', type=str, default="g1")
    parser.add_argument('--record_video', action='store_true')
    args = parser.parse_args()
    
    # ⚠️ 请确认 ONNX 路径是否正确
    checkpoint = "/home/ubuntu/Desktop/robot_lab/logs/rsl_rl/unitree_g1_MotionTracking_flat/2026-02-24_01-37-17_253_MOEMLP/exported/MOEMLPActorCritic.onnx"
    motion_file = "/home/ubuntu/Desktop/robot_lab/source/robot_lab/robot_lab/tasks/manager_based/MotionTracking/config/g1/motion/up_down/113_08_poses.npz"
    
    assert os.path.exists(checkpoint), f"Policy path {checkpoint} does not exist!"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = HumanoidEnv(policy_path=checkpoint, motion_path=motion_file, robot_type=args.robot, device=device, record_video=args.record_video)
    env.run()