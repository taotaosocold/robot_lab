#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from collections import deque

# ========== 路径配置 ==========
network_script_directory = "/home/ubuntu/Desktop/robot_lab/source/robot_lab/robot_lab/tasks/manager_based/MotionTracking/config/g1/agents"
if network_script_directory not in sys.path:
    sys.path.append(network_script_directory)

from Transformer_ActorCritic import MOEMLPActorCritic
import utils.math_utils as math_utils


# ========== 自定义 Wrapper：返回 action + gate_weights ==========
class MOEMLPAnalyzerWrapper(torch.nn.Module):
    """
    包装 MOEMLPActorCritic，推理时同时返回 action 和 gate_weights
    用于分析门控行为，不参与训练
    """
    def __init__(self, original_model, use_future=False, future_steps=1, future_dim=0):
        super().__init__()
        self.actor = original_model
        self.use_future = use_future
        self.future_steps = future_steps
        self.future_dim = future_dim
        
    def forward(self, proprio_history: torch.Tensor):
        """
        输入: proprio_history [B, T, proprio_dim]
        返回: (actions [B, num_actions], gate_weights [B, num_experts])
        """
        # 1. 获取 proprio 并归一化（复用模型内部逻辑）
        proprio = proprio_history[:, -1, :]  # 取最后一步，同 _get_proprio
        proprio_norm = self.actor.actor_proprio_norm(proprio)
        
        # 2. 计算门控权重
        gate_logits = self.actor.gate_mlp(proprio_norm)
        gate_weights = F.softmax(gate_logits, dim=-1)  # [B, num_experts]
        
        # 3. 计算专家输出并加权融合
        expert_outs = torch.stack([exp(proprio_norm) for exp in self.actor.experts], dim=1)  # [B, num_experts, num_actions]
        action_mean = torch.bmm(gate_weights.unsqueeze(1), expert_outs).squeeze(1)  # [B, num_actions]
        
        return action_mean, gate_weights


# ========== 评估环境类 ==========
class MOEMLPEvaluator:
    def __init__(self, checkpoint_path, motion_path, robot_type="g1", device="cuda"):
        self.robot_type = robot_type
        self.device = device
        self.motion_path = motion_path
        self.checkpoint_path = checkpoint_path

        self.load_motion()
        self.load_model()
        
        # 机器人配置（与训练/导出时一致）
        self.mujoco2isaac_dof_index = torch.tensor(
            [0, 6, 12, 1, 7, 13, 18, 2, 8, 14, 19, 3, 9, 15, 20, 4, 10, 16, 21, 5, 11, 17, 22], 
            device=self.device
        )
        self.isaac2mujoco_dof_index = torch.tensor(
            [0, 3, 7, 11, 15, 19, 1, 4, 8, 12, 16, 20, 2, 5, 9, 13, 17, 21, 6, 10, 14, 18, 22], 
            device=self.device
        )
        self.motion_body_indexes = torch.tensor([0, 5, 15, 23, 6, 16, 24, 4, 13, 21, 25, 14, 22, 26], device=self.device)
        self.robot_body_indexes = torch.tensor([1, 3, 5, 7, 9, 11, 13, 14, 16, 18, 19, 21, 23, 24], device=self.device)
        self.anchor_index = 4
        
        if robot_type == "g1":
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
        
        # 仿真参数
        self.sim_duration = 60.0
        self.sim_dt = 0.005
        self.sim_decimation = 4
        self.control_dt = self.sim_dt * self.sim_decimation
        
        # 分析用缓冲区
        self.last_action = torch.zeros(self.num_actions, device=self.device, dtype=torch.float32)
        self.history_length = 1
        self.proprio_history_buffer = deque(maxlen=self.history_length)
        for _ in range(self.history_length):
            self.proprio_history_buffer.append(torch.zeros(self.policy_proprio_dim, device=self.device, dtype=torch.float32))
        
        # 门控权重记录
        self.gate_weights_history = []  # List of [num_experts]

    def load_motion(self):
        """加载参考运动数据"""
        print(f"Loading motion from {self.motion_path}")
        data = np.load(self.motion_path, allow_pickle=True)
        self.joint_pos = torch.tensor(data['joint_pos'], device=self.device, dtype=torch.float32)
        self.joint_vel = torch.tensor(data['joint_vel'], device=self.device, dtype=torch.float32)
        self.body_pos_w = torch.tensor(data['body_pos_w'], device=self.device, dtype=torch.float32)
        self.body_quat_w = torch.tensor(data['body_quat_w'], device=self.device, dtype=torch.float32)
        self.body_lin_vel_w = torch.tensor(data['body_lin_vel_w'], device=self.device, dtype=torch.float32)
        self.body_ang_vel_w = torch.tensor(data['body_ang_vel_w'], device=self.device, dtype=torch.float32)
        self.motion_len = self.joint_pos.shape[0]
        print(f"✓ Motion loaded: {self.motion_len} frames")

    def load_model(self):
        """加载 PyTorch 模型 checkpoint"""
        print(f"Loading PyTorch checkpoint from {self.checkpoint_path}")
        
        # ========== 模型配置（必须与训练时一致！）==========
        self.future_steps = 20
        self.history_length = 1
        self.num_actions = 23
        self.num_experts = 15  # ⚠️ 根据你的训练配置修改
        self.policy_proprio_dim = 253
        self.critic_proprio_dim = 334
        self.future_dim = 256
        self.mlp_hidden_dims = [512, 256, 128]
        self.gate_mlp_hidden_dims = [512, 256, 128]
        self.activation = "ELU"
        self.init_noise_std = 1.0
        self.obs_groups = {
            "policy": ["policy_proprio_history", "future_motion"],
            "critic": ["critic_proprio_history", "future_motion"],
        }
        # ================================================
        
        # 构建 dummy obs 初始化模型
        dummy_obs = {
            "policy_proprio_history": torch.zeros(1, self.history_length, self.policy_proprio_dim),
            "critic_proprio_history": torch.zeros(1, self.history_length, self.critic_proprio_dim),
            "future_motion": torch.zeros(1, self.future_steps, self.future_dim),
        }
        
        # 初始化模型
        self.model = MOEMLPActorCritic(
            obs=dummy_obs,
            obs_groups=self.obs_groups,
            num_actions=self.num_actions,
            num_experts=self.num_experts,
            mlp_hidden_dims=self.mlp_hidden_dims,
            gate_mlp_hidden_dims=self.gate_mlp_hidden_dims,
            init_noise_std=self.init_noise_std,
            activation=self.activation,
        ).to(self.device)
        
        # 加载权重
        state_dict = torch.load(self.checkpoint_path, map_location=self.device)
        load_dict = state_dict.get('model_state_dict', state_dict)
        self.model.load_state_dict(load_dict, strict=True)
        self.model.eval()
        
        # 冻结归一化层
        for module in self.model.modules():
            if hasattr(module, 'eval'):
                module.eval()
        
        # 包装为分析用 wrapper
        self.analyzer = MOEMLPAnalyzerWrapper(
            self.model,
            use_future=False,  # 分析时不使用 future_motion
            future_steps=self.future_steps,
            future_dim=self.future_dim
        ).to(self.device)
        self.analyzer.eval()
        
        print(f"✓ Model loaded: {self.num_experts} experts, proprio_dim={self.policy_proprio_dim}")

    def get_proprio_obs(self, curr_timestep):
        """构建 proprio 观测（与原代码逻辑一致）"""
        # 注意：这里用 dummy 状态模拟，实际评估时如果需要真实机器人状态，需替换为 mujoco/isaac 接口
        # 为简化，这里直接用 motion 数据 + 默认状态构造 proprio
        
        # 从 motion 获取参考状态
        target_idx = min(curr_timestep, self.motion_len - 1)
        motion_joint_pos = self.joint_pos[target_idx]
        motion_joint_vel = self.joint_vel[target_idx]
        command = torch.cat([motion_joint_pos, motion_joint_vel], dim=-1)  # 46
        
        # Dummy 机器人状态（评估时如需真实状态，请替换此处）
        anchor_pos_w = torch.zeros(3, device=self.device)
        anchor_quat_w = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)  # identity quat
        
        # 简化版 proprio（与训练时拼接逻辑一致）
        motion_anchor_ori_b = torch.zeros(6, device=self.device)  # 简化
        robot_body_pos_r = torch.zeros(42, device=self.device)    # 14 bodies * 3
        robot_body_ori_r = torch.zeros(84, device=self.device)    # 14 bodies * 6
        base_lin_vel = torch.zeros(3, device=self.device)
        base_ang_vel = torch.zeros(3, device=self.device)
        dof_pos = torch.zeros(self.num_dofs, device=self.device)
        dof_vel = torch.zeros(self.num_dofs, device=self.device)
        
        proprio_obs = torch.cat([
            command,                    # 46
            motion_anchor_ori_b,        # 6
            robot_body_pos_r,           # 42
            robot_body_ori_r,           # 84
            base_lin_vel,               # 3
            base_ang_vel,               # 3
            dof_pos,                    # 23
            dof_vel,                    # 23
            self.last_action,           # 23
        ], dim=-1)
        
        return proprio_obs

    def run_evaluation(self, save_gate_path=None):
        """执行评估并记录门控权重"""
        print(f"\n🚀 Starting evaluation on {self.motion_path}")
        print(f"   Motion frames: {self.motion_len}, Control freq: {1/self.control_dt:.1f}Hz")
        
        total_steps = int(self.sim_duration / self.control_dt)
        eval_steps = min(total_steps, self.motion_len)  # 不超过 motion 长度
        
        gate_weights_all = []  # [T, num_experts]
        
        with torch.no_grad():
            for step in tqdm(range(eval_steps), desc="Evaluating"):
                # 1. 构建 proprio 观测
                proprio_obs = self.get_proprio_obs(step)
                self.proprio_history_buffer.append(proprio_obs)
                
                # 2. 堆叠历史 [B=1, T, dim]
                proprio_history = torch.stack(list(self.proprio_history_buffer), dim=0).unsqueeze(0)
                
                # 3. 推理获取 action + gate_weights
                action, gate_weights = self.analyzer(proprio_history)  # gate_weights: [1, num_experts]
                
                # 4. 记录门控权重
                gate_weights_cpu = gate_weights.squeeze(0).cpu().numpy()
                gate_weights_all.append(gate_weights_cpu)
                
                # 5. 更新 last_action（用于下一帧 proprio）
                self.last_action = action.squeeze(0).clone()
        
        # ========== 统计并输出结果 ==========
        gate_weights_all = np.array(gate_weights_all)  # [T, num_experts]
        self._print_gate_statistics(gate_weights_all)
        
        # 可选：保存门控权重到文件
        if save_gate_path:
            os.makedirs(os.path.dirname(save_gate_path), exist_ok=True)
            np.savez(save_gate_path, 
                     gate_weights=gate_weights_all,
                     motion_name=os.path.basename(self.motion_path),
                     num_experts=self.num_experts)
            print(f"✓ Gate weights saved to {save_gate_path}")
        
        return gate_weights_all

    def _print_gate_statistics(self, gate_weights: np.ndarray):
        """打印门控权重统计信息"""
        T, num_experts = gate_weights.shape
        
        print(f"\n{'='*60}")
        print(f"📊 Gate Weights Statistics")
        print(f"{'='*60}")
        print(f"Total timesteps: {T}")
        print(f"Number of experts: {num_experts}")
        print(f"-"*60)
        
        # 1. 平均权重
        mean_weights = gate_weights.mean(axis=0)
        print(f"\n📈 Average gate weights per expert:")
        for i, w in enumerate(mean_weights):
            print(f"   Expert {i:2d}: {w:6.4f}  ({w*100:5.2f}%)")
        
        # 2. 主导专家分析
        dominant_expert = np.argmax(gate_weights, axis=1)  # [T]
        unique, counts = np.unique(dominant_expert, return_counts=True)
        print(f"\n🏆 Dominant expert per timestep:")
        for exp_id, count in zip(unique, counts):
            pct = count / T * 100
            print(f"   Expert {exp_id:2d}: {count:4d} timesteps  ({pct:5.2f}%)")
        
        # 3. 熵分析（衡量门控的"确定性"）
        # H = -sum(p * log(p)), 越低表示门控越"专一"
        eps = 1e-8
        entropy = -np.sum(gate_weights * np.log(gate_weights + eps), axis=1)  # [T]
        print(f"\n🎯 Gate entropy (lower = more decisive):")
        print(f"   Mean: {entropy.mean():.4f}")
        print(f"   Std:  {entropy.std():.4f}")
        print(f"   Min:  {entropy.min():.4f}  (most decisive)")
        print(f"   Max:  {entropy.max():.4f}  (most uncertain)")
        
        # 4. 专家激活频率（权重 > 0.1 视为"激活"）
        threshold = 0.1
        activation_counts = (gate_weights > threshold).sum(axis=0)
        print(f"\n⚡ Expert activation frequency (weight > {threshold}):")
        for i, cnt in enumerate(activation_counts):
            pct = cnt / T * 100
            print(f"   Expert {i:2d}: {cnt:4d}/{T}  ({pct:5.2f}%)")
        
        print(f"{'='*60}\n")


# ========== 主函数 ==========
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MOEMLP Gate Weights Analyzer")
    parser.add_argument('--checkpoint', type=str, required=True, 
                        help='Path to PyTorch checkpoint (.pt)')
    parser.add_argument('--motion', type=str, required=True,
                        help='Path to motion file (.npz)')
    parser.add_argument('--robot', type=str, default='g1', choices=['g1'],
                        help='Robot type')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device: cuda or cpu')
    parser.add_argument('--save-gate', type=str, default=None,
                        help='Optional: save gate weights to .npz file')
    
    args = parser.parse_args()
    
    # 检查文件存在
    assert os.path.exists(args.checkpoint), f"Checkpoint not found: {args.checkpoint}"
    assert os.path.exists(args.motion), f"Motion file not found: {args.motion}"
    
    # 设备设置
    device = args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 创建评估器并运行
    evaluator = MOEMLPEvaluator(
        checkpoint_path=args.checkpoint,
        motion_path=args.motion,
        robot_type=args.robot,
        device=device
    )
    
    evaluator.run_evaluation(save_gate_path=args.save_gate)