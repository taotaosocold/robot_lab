import os
import sys
import time
import traceback
import numpy as np
import mujoco
import torch
import pandas as pd
from tqdm import tqdm
from collections import deque
import onnxruntime as ort  # 添加 onnxruntime 导入

# 添加网络脚本路径
network_script_directory = "/home/ubuntu/Desktop/robot_lab/source/robot_lab/robot_lab/tasks/manager_based/MotionTracking/config/g1/agents"
if network_script_directory not in sys.path:
    sys.path.append(network_script_directory)

# 由于改用 ONNX，不再需要导入 PyTorch 模型类，但保留路径设置以防万一
# from Transformer_ActorCritic import (...) 
# from tensordict import TensorDict  # 不再需要 TensorDict
import utils.math_utils as math_utils


# ==================== 配置区域（直接在此修改路径）====================
# 更新为 ONNX 模型路径 (参考 mujoco_onnx_MT.py)
CHECKPOINT_PATH = "/home/ubuntu/Desktop/robot_lab/logs/rsl_rl/unitree_g1_MotionTracking_flat/2026-02-25_13-49-26/exported/policy.onnx"
MOTION_FOLDER = "/home/ubuntu/Desktop/hjq/gmr_npz/choose"
OUTPUT_CSV = os.path.dirname(CHECKPOINT_PATH) + "/evaluate_mujoco.csv"
ROBOT_TYPE = "g1"
DEVICE = "cuda"
# 【新增】根位置偏差阈值，超过则判定该运动序列失败
ROOT_POS_THRESHOLD = 0.35  # 单位：米
#===================================================================

class HumanoidEnv:
    def __init__(self, policy_path, robot_type="g1", device="cuda"):
        self.robot_type = robot_type
        self.device = device
        self.policy_path = policy_path
        # 加载模型配置
        self.load_model()
        
        # MuJoCo 与 Isaac Lab 的 DOF 索引映射
        self.mujoco2isaac_dof_index = torch.tensor(
            [0, 6, 12, 1, 7, 13, 18, 2, 8, 14, 19, 3, 9, 15,  20, 4, 10, 16, 21, 5, 11, 17, 22], 
            device=self.device
        )
        self.isaac2mujoco_dof_index = torch.tensor(
            [0, 3, 7, 11, 15, 19, 1, 4, 8, 12, 16, 20, 2, 5, 9, 13 , 17, 21, 6, 10, 14, 18, 22], 
            device=self.device
        )
        
        # 身体部位索引（用于计算 MPKPE）
        self.motion_body_indexes = torch.tensor(
            [0, 5, 15, 23, 6, 16, 24, 4, 13,  21, 25, 14, 22, 26], 
            device=self.device
        )
        self.robot_body_indexes = torch.tensor(
            [1, 3, 5, 7, 9, 11, 13, 14, 16, 18, 19, 21, 23, 24], 
            device=self.device
        )
        
        # 锚点索引（torso）
        self.anchor_index = 4
        
        if robot_type == "g1":
            model_path = "/home/ubuntu/Desktop/hjq/assets/g1_23dof_rev_1_0.xml"
            
            # 关节参数
            self.stiffness = np.array([ 
                40.179, 99.098, 40.179, 99.098, 28.501, 28.501,
                40.179, 99.098, 40.179, 99.098, 28.501, 28.501,
                40.179,
                14.25, 14.25, 14.25, 14.25, 14.25,
                14.25, 14.25, 14.25, 14.25, 14.25,
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
                88.0, 139.0, 88.0, 139.0 , 35.0, 35.0,
                88.0,
                25.0, 25.0, 25.0, 25.0, 25.0,
                25.0, 25.0, 25.0, 25.0, 25.0
            ])

            self.default_dof_pos = torch.tensor([
                 -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
                -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
                0.0,
                0.2, 0.2, 0.0, 0.6, 0.0,
                0.2, -0.2, 0.0, 0.6,  0.0,
            ], device=self.device, dtype=torch.float32)

            self.action_scale = torch.tensor([
                0.547, 0.350, 0.547, 0.350, 0.307, 0.307,
                0.547, 0.350,  0.547, 0.350, 0.307, 0.307,
                0.547,
                0.438, 0.438, 0.438, 0.438, 0.438,
                0.438, 0.438, 0.438, 0.438, 0.438
            ], device=self.device, dtype=torch.float32)

            self.num_actions = 23
            self.num_dofs = 23
        else:
            raise ValueError(f"Robot type {robot_type} not supported! ")
        
        # 仿真参数
        self.sim_dt = 0.005
        self.sim_decimation = 4
        self.control_dt = self.sim_dt * self.sim_decimation
        
        # 加载 MuJoCo 模型
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = self.sim_dt
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_step(self.model, self.data)
        
        self.last_action = torch.zeros(self.num_actions, device=self.device, dtype=torch.float32)
        
        # 历史缓冲
        self.proprio_history_buffer = deque(maxlen=self.history_length)
        for _ in range(self.history_length):
            self.proprio_history_buffer.append(
                torch.zeros(self.policy_proprio_dim, device=self.device, dtype=torch.float32)
            )

    def load_model(self):
        """加载 ONNX 策略模型 (与 mujoco_onnx_MT.py 对齐)"""
        print(f"Loading ONNX policy from {self.policy_path}")
        
        # 基础参数依然需要，用于维度对齐
        self.future_steps = 20
        self.history_length = 1
        self.num_actions = 23
        # 根据 get_proprio_obs 的实际 concatenation 逻辑，维度应为 253
        # 46(command) + 6(ori_b) + 42(pos_r) + 84(ori_r) + 3(lin_vel) + 3(ang_vel) + 23(dof_pos) + 23(dof_vel) + 23(last_action) = 253
        self.policy_proprio_dim = 253 
        
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == "cuda" else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(self.policy_path, providers=providers)
        
        # 获取输入节点的名称
        inputs = self.session.get_inputs()
        # 假设输入顺序与 mujoco_onnx_MT.py 一致：future_motion, policy_proprio_history
        # 为了安全，通过名称匹配
        for inp in inputs:
            if 'future' in inp.name.lower():
                self.input_name_future = inp.name
            elif 'proprio' in inp.name.lower():
                self.input_name_proprio = inp.name
        
        # 如果未能自动匹配，使用默认名称（根据 mujoco_onnx_MT.py）
        if not hasattr(self, 'input_name_future'):
            self.input_name_future = inputs[0].name
        if not hasattr(self, 'input_name_proprio'):
            self.input_name_proprio = inputs[1].name

    def load_motion(self, motion_path):
        """加载运动数据"""
        data = np.load(motion_path, allow_pickle=True)
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
        self.motion_path = motion_path

    def align_motion_to_robot(self):
        """将运动数据对齐到机器人初始位姿"""
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
        """获取未来运动观测 (与 mujoco_onnx_MT.py 完全对齐)"""
        target_indices = self.tar_obs_steps + curr_timestep
        target_indices = torch.clamp(target_indices, max=self.motion_len - 1)

        body_pos_w = self.body_pos_w[target_indices][:, self.motion_body_indexes]
        body_quat_w = self.body_quat_w[target_indices][:, self.motion_body_indexes]
        body_lin_vel_w = self.body_lin_vel_w[target_indices][:, self.motion_body_indexes]
        body_ang_vel_w = self.body_ang_vel_w[target_indices][:, self.motion_body_indexes]
        joint_pos = self.joint_pos[target_indices]
        joint_vel = self.joint_vel[target_indices]

        curr_anchor_quat_w = self.body_quat_w[curr_timestep, self.anchor_index]
        heading_quat = math_utils.yaw_quat(curr_anchor_quat_w.unsqueeze(0))
        heading_inv = math_utils.quat_conjugate(heading_quat)

        curr_anchor_pos_w = self.body_pos_w[curr_timestep, self.anchor_index]
        diff_pos = body_pos_w - curr_anchor_pos_w.view(1, 1, 3)

        T, B, _ = diff_pos.shape
        heading_inv = heading_inv.view(1, 1, 4).expand(T, B, 4).reshape(-1, 4)

        motion_body_pos = math_utils.quat_apply(heading_inv, diff_pos.reshape(-1, 3)).view(T, B, 3)
        motion_body_quat = math_utils.quat_mul(heading_inv, body_quat_w.reshape(-1, 4))
        motion_body_ori = math_utils.matrix_from_quat(motion_body_quat)[..., :2].reshape(T, B, 6)
        motion_body_lin_vel = math_utils.quat_apply(heading_inv, body_lin_vel_w.reshape(-1, 3)).view(T, B, 3)
        motion_body_ang_vel = math_utils.quat_apply(heading_inv, body_ang_vel_w.reshape(-1, 3)).view(T, B, 3)

        future_obs = torch.cat([
            motion_body_pos.reshape(T, -1),
            motion_body_ori.reshape(T, -1),
            motion_body_lin_vel.reshape(T, -1),
            motion_body_ang_vel.reshape(T, -1),
            joint_pos.reshape(T, -1),
            joint_vel.reshape(T, -1)
        ], dim=-1)
        
        return future_obs

    def get_proprio_obs(self, curr_timestep):
        """获取本体观测 (与 mujoco_onnx_MT.py 完全对齐)"""
        dof_pos = torch.from_numpy(self.data.qpos[-self.num_dofs:].astype(np.float32)).to(self.device)
        dof_vel = torch.from_numpy(self.data.qvel[-self.num_dofs:].astype(np.float32)).to(self.device)
        anchor_pos_w = torch.from_numpy(self.data.body('torso_link').xpos.astype(np.float32)).to(self.device)
        anchor_quat_w = torch.from_numpy(self.data.body('torso_link').xquat.astype(np.float32)).to(self.device)
        
        base_lin_vel = torch.from_numpy(self.data.qvel.astype(np.float32)[0:3]).to(self.device)
        base_ang_vel = torch.from_numpy(self.data.qvel.astype(np.float32)[3:6]).to(self.device)
        gravity_vec_w = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=torch.float32)
        projected_gravity_b = math_utils.quat_apply_inverse(anchor_quat_w, gravity_vec_w)
         
        # compute motion anchor pos/ori_b
        target_idx = min(curr_timestep, self.motion_len - 1)
        motion_joint_pos = self.joint_pos[target_idx]
        motion_joint_vel = self.joint_vel[target_idx]
        command = torch.cat([motion_joint_pos, motion_joint_vel], dim=-1)
        
        rel_pos_w = self.body_pos_w[target_idx, self.anchor_index] - anchor_pos_w
        motion_anchor_pos_b = math_utils.quat_apply_inverse(anchor_quat_w, rel_pos_w)
        rel_quat_b = math_utils.quat_mul(math_utils.quat_conjugate(anchor_quat_w), self.body_quat_w[target_idx, self.anchor_index])
        motion_anchor_ori_b = math_utils.matrix_from_quat(rel_quat_b)[..., :2].reshape(-1)

        # compute robot body pos/ori_r
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
            command,              # 46
            motion_anchor_ori_b,  # 6
            robot_body_pos_r,     # 42
            robot_body_ori_r,     # 84
            base_lin_vel,         # 3
            base_ang_vel,         # 3
            dof_pos,              # 23
            dof_vel,              # 23
            self.last_action,     # 23
        ], dim=-1)
        
        return proprio_obs

    def compute_metrics(self, curr_timestep):
        """计算当前帧的评估指标（保留你的测量逻辑不变）"""
        target_idx = min(curr_timestep, self.motion_len - 1)
        
        # 获取机器人当前状态
        robot_dof_pos = torch.from_numpy(self.data.qpos[-self.num_dofs:].astype(np.float32)).to(self.device)
        robot_dof_pos = (robot_dof_pos - self.default_dof_pos)[self.mujoco2isaac_dof_index]
        
        robot_body_pos_w = torch.from_numpy(self.data.xpos.astype(np.float32)).to(self.device)[self.robot_body_indexes]
        robot_body_quat_w = torch.from_numpy(self.data.xquat.astype(np.float32)).to(self.device)[self.robot_body_indexes]
        
        anchor_pos_w = torch.from_numpy(self.data.body('torso_link').xpos.astype(np.float32)).to(self.device)
        anchor_quat_w = torch.from_numpy(self.data.body('torso_link').xquat.astype(np.float32)).to(self.device)
        
        # 目标运动状态
        target_joint_pos = self.joint_pos[target_idx]
        target_body_pos_w = self.body_pos_w[target_idx][self.motion_body_indexes]
        target_body_lin_vel_w = self.body_lin_vel_w[target_idx][self.motion_body_indexes]
        target_body_ang_vel_w = self.body_ang_vel_w[target_idx][self.motion_body_indexes]
        target_anchor_lin_vel = self.body_lin_vel_w[target_idx, self.anchor_index]
        target_anchor_ang_vel = self.body_ang_vel_w[target_idx, self.anchor_index]
        
        # ========== 【新增】Root Position Error (m) ==========
        target_anchor_pos_w = self.body_pos_w[target_idx, self.anchor_index]
        root_pos_err = torch.norm(anchor_pos_w - target_anchor_pos_w).cpu().item()
        
        # ========== MPJPE: 关节位置误差 (rad) ==========
        joint_pos_err = torch.abs(robot_dof_pos - target_joint_pos)
        mpjpe = joint_pos_err.mean().cpu().item()
        
        # ========== MPKPE: 关键点位置误差 (m) ==========
        anchor_quat_inv = math_utils.quat_conjugate(anchor_quat_w).view(1, 4).expand(len(self.robot_body_indexes), 4)
        robot_body_pos_r = math_utils.quat_apply(anchor_quat_inv, (robot_body_pos_w - anchor_pos_w.view(1, 3))).reshape(-1, 3)
        target_body_pos_r = math_utils.quat_apply(anchor_quat_inv, (target_body_pos_w - anchor_pos_w.view(1, 3))).reshape(-1, 3)
        
        keypoint_pos_err = torch.norm(robot_body_pos_r - target_body_pos_r, dim=-1)
        mpkpe = keypoint_pos_err.mean().cpu().item()
        
        # ========== Anchor Linear Velocity Error (m/s) ==========
        robot_anchor_lin_vel = torch.from_numpy(self.data.sensor('imu-torso-linear-velocity').data.astype(np.float32)).to(self.device)
        robot_anchor_lin_vel_r = math_utils.quat_apply_inverse(anchor_quat_w, robot_anchor_lin_vel)
        target_anchor_lin_vel_r = math_utils.quat_apply_inverse(anchor_quat_w, target_anchor_lin_vel)
        lin_vel_err = torch.norm(robot_anchor_lin_vel_r - target_anchor_lin_vel_r).cpu().item()
        
        # ========== Anchor Angular Velocity Error (rad/s) ==========
        robot_anchor_ang_vel = torch.from_numpy(self.data.sensor('imu-torso-angular-velocity').data.astype(np.float32)).to(self.device)
        robot_anchor_ang_vel_r = math_utils.quat_apply_inverse(anchor_quat_w, robot_anchor_ang_vel)
        target_anchor_ang_vel_r = math_utils.quat_apply_inverse(anchor_quat_w, target_anchor_ang_vel)
        ang_vel_err = torch.norm(robot_anchor_ang_vel_r - target_anchor_ang_vel_r).cpu().item()
        
        # 【修改】返回结果中添加 root_pos_err
        return mpjpe, mpkpe, lin_vel_err, ang_vel_err, root_pos_err

    def evaluate_motion(self, motion_path):
        """评估单个运动序列"""
        # 加载运动数据
        self.load_motion(motion_path)
        self.align_motion_to_robot()
        
        # 重置仿真
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_step(self.model, self.data)
        
        # 重置历史缓冲
        self.proprio_history_buffer.clear()
        for _ in range(self.history_length):
            self.proprio_history_buffer.append(
                torch.zeros(self.policy_proprio_dim, device=self.device, dtype=torch.float32)
            )
        self.last_action = torch.zeros(self.num_actions, device=self.device, dtype=torch.float32)
        
        # 预先初始化 self.pd_target 以防第一步之前使用导致的错误
        self.pd_target = np.zeros(self.num_actions)
        
        # 统计指标
        total_mpjpe = 0.0
        total_mpkpe = 0.0
        total_lin_vel_err =  0.0
        total_ang_vel_err = 0.0
        total_root_pos_err = 0.0
        valid_frames = 0
        root_pos_exceeded = False  # 【新增】标记根位置偏差是否超过阈值
        
        # 运行仿真并收集指标
        num_steps = int(self.motion_len * self.sim_decimation)
        # 提前分配目标步长张量，供 get_future_obs 使用
        self.tar_obs_steps = torch.arange(1, self.future_steps + 1, device=self.device, dtype=torch.int)
        
        for i in range(num_steps):
            curr_timestep = i // self.sim_decimation
            
            # 获取观测
            proprio_obs = self.get_proprio_obs(curr_timestep)
            
            if i % self.sim_decimation == 0:
                future_obs = self.get_future_obs(curr_timestep).unsqueeze(0)
                self.proprio_history_buffer.append(proprio_obs)
                policy_proprio_history = torch.stack(list(self.proprio_history_buffer), dim=0).unsqueeze(0)
                
                # 【修改】：使用 ONNX 推理
                policy_proprio_history_np = policy_proprio_history.cpu().numpy()
                future_obs_np = future_obs.cpu().numpy()

                inputs = {
                    self.input_name_future: future_obs_np,
                    self.input_name_proprio: policy_proprio_history_np,
                }
                
                ort_outputs = self.session.run(None, inputs)
                action_np = ort_outputs[0].squeeze()
                action = torch.from_numpy(action_np).to(self.device)
                
                self.last_action = action.clone()
                
                # 使用与 mujoco_onnx_MT.py 完全一致的 PD 控制目标计算
                pd_target_tensor = action * self.action_scale[self.mujoco2isaac_dof_index] + self.default_dof_pos[self.mujoco2isaac_dof_index]
                self.pd_target = pd_target_tensor[self.isaac2mujoco_dof_index].cpu().numpy()
            
            # PD 控制
            dof_pos = self.data.qpos[-self.num_dofs:]
            dof_vel = self.data.qvel[-self.num_dofs:]
            torque = (self.pd_target - dof_pos) * self.stiffness - dof_vel * self.damping
            torque = np.clip(torque, -self.torque_limits, self.torque_limits)
            self.data.ctrl = torque
            
            mujoco.mj_step(self.model, self.data)
             
            # 计算指标（只在控制步计算）
            if i % self.sim_decimation == 0 and curr_timestep < self.motion_len:
                # 【修改】接收新增的 root_pos_err 返回值
                mpjpe, mpkpe, lin_vel_err, ang_vel_err, root_pos_err = self.compute_metrics(curr_timestep)
                total_mpjpe += mpjpe
                total_mpkpe += mpkpe
                total_lin_vel_err += lin_vel_err
                total_ang_vel_err += ang_vel_err
                total_root_pos_err += root_pos_err
                valid_frames += 1
                
                # 【新增】检查根位置偏差是否超过阈值
                if root_pos_err > ROOT_POS_THRESHOLD:
                    root_pos_exceeded = True
        
        # 计算平均值
        if valid_frames > 0:
            avg_mpjpe = total_mpjpe / valid_frames
            avg_mpkpe = total_mpkpe / valid_frames
            avg_lin_vel_err = total_lin_vel_err / valid_frames
            avg_ang_vel_err = total_ang_vel_err / valid_frames
            avg_root_pos_err = total_root_pos_err / valid_frames
        else:
            avg_mpjpe = avg_mpkpe = avg_lin_vel_err = avg_ang_vel_err = avg_root_pos_err = 0.0
        
        # 【修改】返回结果中添加 root_pos_err 和基于根位置的成功标志
        return {
            'mpjpe': avg_mpjpe,
            'mpkpe': avg_mpkpe,
            'lin_vel_err': avg_lin_vel_err,
            'ang_vel_err': avg_ang_vel_err,
            'root_pos_err': avg_root_pos_err,  # 【新增】平均根位置误差
            'success': not root_pos_exceeded,  # 【修改】根位置偏差从未超过阈值则成功
            'frames': valid_frames
        }

    def close(self):
        """清理资源"""
        pass

def main():
    # 验证路径
    assert os.path.exists(CHECKPOINT_PATH), f"Policy path {CHECKPOINT_PATH} does not exist!"
    assert os.path.exists(MOTION_FOLDER), f"Motion folder {MOTION_FOLDER} does not exist!"
    # 获取所有 .npz 文件
    motion_files = sorted([
        os.path.join(MOTION_FOLDER, f) 
        for f in os.listdir(MOTION_FOLDER) 
        if f.endswith('.npz')
    ])

    if len(motion_files) == 0:
        print(f"[ERROR] No .npz files found in {MOTION_FOLDER}!")
        return

    print(f"[INFO] Found {len(motion_files)} motion files to evaluate")
    print(f"[INFO] Checkpoint: {CHECKPOINT_PATH}")
    print(f"[INFO] Robot: {ROBOT_TYPE}")
    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] Root Position Threshold: {ROOT_POS_THRESHOLD} m")  # 【新增】打印阈值

    # 初始化环境
    env = HumanoidEnv(policy_path=CHECKPOINT_PATH, robot_type=ROBOT_TYPE, device=DEVICE)

    # 评估所有运动
    results = []
    print(f"\n{'='*80}")
    print(f"Starting Evaluation ({len(motion_files)} motions)")
    print(f"{'='*80}\n")

    for idx, motion_path in enumerate(tqdm(motion_files, desc="Evaluating motions")):
        motion_name = os.path.basename(motion_path).split('.')[0]
        
        try:
            metrics = env.evaluate_motion(motion_path)
            results.append({
                'rank': idx + 1,
                'motion_name': motion_name,
                'mpjpe_rad': metrics['mpjpe'],
                'mpkpe_m': metrics['mpkpe'],
                'anchor_lin_vel_err_m_s': metrics['lin_vel_err'],
                'anchor_ang_vel_err_rad_s': metrics['ang_vel_err'],
                'root_pos_err_m': metrics['root_pos_err'],  # 【新增】根位置误差
                'success': "Success" if metrics['success'] else "Failed",  # 【修改】基于根位置偏差判断
                'frames': metrics['frames']
            })
        except Exception as e:
            print(f"[WARNING] Failed to evaluate {motion_name}: {e}")
            traceback.print_exc()
            results.append({
                'rank': idx + 1,
                'motion_name': motion_name,
                'mpjpe_rad': 0.0,
                'mpkpe_m': 0.0,
                'anchor_lin_vel_err_m_s': 0.0,
                'anchor_ang_vel_err_rad_s': 0.0,
                'root_pos_err_m': 0.0,  # 【新增】
                'success': "Failed",
                'frames': 0
            })

    env.close()

    # ========== 最终报告 ==========
    print(f"\n{'='*80}")
    print(f"Final Evaluation Result ({len(results)} motions)")
    print(f"{'='*80}")

    # 计算统计
    successful_results = [r for r in results if r['success'] == "Success"]
    if len(successful_results) > 0:
        mpjpe_values = np.array([r['mpjpe_rad'] for r in successful_results])
        mpkpe_values = np.array([r['mpkpe_m'] for r in successful_results])
        lin_vel_values = np.array([r['anchor_lin_vel_err_m_s'] for r in successful_results])
        ang_vel_values = np.array([r['anchor_ang_vel_err_rad_s'] for r in successful_results])
        root_pos_values = np.array([r['root_pos_err_m'] for r in successful_results])  # 【新增】
        
        print(f"Avg MPJPE: {mpjpe_values.mean():.5f} rad (±{mpjpe_values.std():.5f})")
        print(f"Avg MPKPE: {mpkpe_values.mean():.5f} m (±{mpkpe_values.std():.5f})")
        print(f"Avg Anchor Linear Vel Error: {lin_vel_values.mean():.5f} m/s (±{lin_vel_values.std():.5f})")
        print(f"Avg Anchor Angular Vel Error: {ang_vel_values.mean():.5f} rad/s (±{ang_vel_values.std():.5f})")
        print(f"Avg Root Position Error: {root_pos_values.mean():.5f} m (±{root_pos_values.std():.5f})")  # 【新增】
        # 【修改】成功率基于根位置偏差判断
        success_rate = len(successful_results) / len(results) * 100
        print(f"Success Rate (root pos err ≤ {ROOT_POS_THRESHOLD}m): {success_rate:.2f}%")
    else:
        print("[ERROR] No successful evaluations!")

    # 详细结果（按 MPJPE 降序排序）
    print(f"\nPer-motion detailed results (sorted by MPJPE descending):")
    # 【修改】表头添加 Root Pos Err 列
    print(f"{'Rank': <4} {'Motion Name': <40} {'MPJPE (rad)': <14} {'MPKPE (m)': <14} {'LinVel Err (m/s)': <18} {'AngVel Err (rad/s)': <20} {'RootPosErr (m)': <16} {'Status'}")
    print("-" * 140)

    sorted_results = sorted(successful_results, key=lambda x: x['mpjpe_rad'], reverse=True)
    for rank, r in enumerate(sorted_results, 1):
        # 【修改】输出添加 root_pos_err_m 列
        print(f"{rank: <4} {r['motion_name']: <40} {r['mpjpe_rad']: <14.5f} {r['mpkpe_m']: <14.5f} {r['anchor_lin_vel_err_m_s']: <18.5f} {r['anchor_ang_vel_err_rad_s']: <20.5f} {r['root_pos_err_m']: <16.5f} {r['success']}")

    # 保存 CSV
    df = pd.DataFrame(sorted_results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDetailed results saved to: {OUTPUT_CSV}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()