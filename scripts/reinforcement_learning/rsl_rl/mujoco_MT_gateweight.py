#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MOETransformer 门控权重分析脚本（最终版 - 属性名完全匹配）
- 直接使用模型内部属性: future_norm, actor_gate_head, actor_experts 等
- 门控计算: _forward_actor_gate(future_motion)
- 输出: 门控权重统计 + 专家激活分析
"""

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

from Transformer_ActorCritic import MOEMLPTransformerEncoderActorCritic
from tensordict import TensorDict
import utils.math_utils as math_utils


# ========== 自定义 Wrapper：直接复用模型内部逻辑 ==========
class MOETransformerAnalyzerWrapper(torch.nn.Module):
    """
    包装 MOEMLPTransformerEncoderActorCritic
    直接调用模型内部的 _forward_actor_gate 和专家融合逻辑
    属性名严格匹配真实代码
    """
    def __init__(self, original_model):
        super().__init__()
        self.actor = original_model  # 必须叫 actor 保持兼容性
        
    def forward(self, proprio_history: torch.Tensor, future_motion: torch.Tensor):
        """
        输入:
            proprio_history: [B, T_p, proprio_dim]
            future_motion: [B, T_f, future_dim]
        返回: (actions [B, num_actions], gate_weights [B, num_experts])
        """
        # ========== 1. 归一化（属性名严格匹配真实代码）==========
        future_norm = self.actor.future_norm(future_motion)  # ✅ future_norm
        proprio_norm = self.actor.actor_proprio_norm(proprio_history)  # ✅ actor_proprio_norm
        
        # ========== 2. 计算门控权重（复用 _forward_actor_gate）==========
        gate_weights = self.actor._forward_actor_gate(future_norm)  # ✅ [B, num_experts]
        
        # ========== 3. 专家融合（复用 actor_experts）==========
        current_proprio = proprio_norm[:, -1, :]  # [B, proprio_dim]
        expert_outs = torch.stack([exp(current_proprio) for exp in self.actor.actor_experts], dim=1)  # [B, num_experts, num_actions]
        action_mean = torch.bmm(gate_weights.unsqueeze(1), expert_outs).squeeze(1)  # [B, num_actions]
        
        return action_mean, gate_weights


# ========== 评估环境类 ==========
class MOETransformerEvaluator:
    def __init__(self, checkpoint_path, motion_path, robot_type="g1", device="cuda"):
        self.robot_type = robot_type
        self.device = device
        self.motion_path = motion_path
        self.checkpoint_path = checkpoint_path

        self.load_motion()
        self.load_model()
        
        # 机器人配置
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
        
        self.sim_duration = 60.0
        self.sim_dt = 0.005
        self.sim_decimation = 4
        self.control_dt = self.sim_dt * self.sim_decimation
        
        self.last_action = torch.zeros(self.num_actions, device=self.device, dtype=torch.float32)
        self.history_length = 1
        self.proprio_history_buffer = deque(maxlen=self.history_length)
        for _ in range(self.history_length):
            self.proprio_history_buffer.append(torch.zeros(self.policy_proprio_dim, device=self.device, dtype=torch.float32))

    def load_motion(self):
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
        print(f"Loading PyTorch checkpoint from {self.checkpoint_path}")
        
        # ========== 模型配置（必须与训练时一致）==========
        self.future_steps = 20
        self.history_length = 1
        self.num_actions = 23
        self.num_experts = 15  # ⚠️ 根据训练配置修改
        
        self.policy_proprio_dim = 253  # ⚠️ 根据训练配置修改
        self.critic_proprio_dim = 334
        num_bodies = 14
        self.future_motion_dim = (num_bodies * 3) + (num_bodies * 6) + (num_bodies * 3) + (num_bodies * 3) + 23 + 23  # = 256
        
        self.policy_cfg = {
            "num_actions": self.num_actions,
            "d_model": 256,
            "nhead": 4,
            "num_encoder_layers": 1,
            "dim_feedforward": 256,
            "num_experts": self.num_experts,
            "mlp_hidden_dims": [512, 256, 128],  # ✅ 直接用配置类参数名
            "activation": "elu", 
            "dropout": 0.1,
            "init_noise_std": 1.0,
        }
        self.obs_groups = {
            "policy": ["policy_proprio_history", "future_motion"],
            "critic": ["critic_proprio_history", "future_motion"],
        }
        # ================================================
        
        dummy_obs = {
            "policy_proprio_history": torch.zeros(1, self.history_length, self.policy_proprio_dim),
            "critic_proprio_history": torch.zeros(1, self.history_length, self.critic_proprio_dim),
            "future_motion": torch.zeros(1, self.future_steps, self.future_motion_dim),
        }
        
        print(f"[DEBUG] Config: mlp_hidden_dims={self.policy_cfg['mlp_hidden_dims']}")
        
        # 初始化模型
        self.model = MOEMLPTransformerEncoderActorCritic(
            obs=dummy_obs,
            obs_groups=self.obs_groups,
            **self.policy_cfg
        ).to(self.device)
        
        # 加载权重
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        
        # 冻结所有模块
        for module in self.model.modules():
            if hasattr(module, 'eval'):
                module.eval()
        
        # [DEBUG] 确认关键属性存在
        print(f"[DEBUG] Checking model attributes:")
        attrs_to_check = [
            'future_norm', 'actor_proprio_norm', 
            'actor_future_proj', 'actor_pos_encoder',
            'actor_gate_transformer', 'actor_gate_head',
            'actor_experts', '_forward_actor_gate'
        ]
        for attr in attrs_to_check:
            exists = hasattr(self.model, attr)
            print(f"   {attr}: {'✅' if exists else '❌'}")
        
        # 包装为分析器
        self.analyzer = MOETransformerAnalyzerWrapper(self.model).to(self.device)
        self.analyzer.eval()
        
        print(f"✓ Model ready: {self.num_experts} experts, proprio_dim={self.policy_proprio_dim}")

    def get_future_obs(self, curr_timestep):
        """构建 future_motion 观测（与原代码完全一致）"""
        target_indices = torch.arange(1, self.future_steps + 1, device=self.device, dtype=torch.int) + curr_timestep
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
        """构建 proprio 观测（简化版，用 motion 数据 + dummy 状态）"""
        target_idx = min(curr_timestep, self.motion_len - 1)
        motion_joint_pos = self.joint_pos[target_idx]
        motion_joint_vel = self.joint_vel[target_idx]
        command = torch.cat([motion_joint_pos, motion_joint_vel], dim=-1)  # 46
        
        # Dummy 状态（如需真实机器人状态，请替换为 mujoco 接口）
        anchor_pos_w = torch.zeros(3, device=self.device)
        anchor_quat_w = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        base_lin_vel = torch.zeros(3, device=self.device)
        base_ang_vel = torch.zeros(3, device=self.device)
        
        motion_anchor_ori_b = torch.zeros(6, device=self.device)
        robot_body_pos_r = torch.zeros(42, device=self.device)
        robot_body_ori_r = torch.zeros(84, device=self.device)
        dof_pos = torch.zeros(self.num_dofs, device=self.device)
        dof_vel = torch.zeros(self.num_dofs, device=self.device)
        
        proprio_obs = torch.cat([
            command, motion_anchor_ori_b, robot_body_pos_r, robot_body_ori_r,
            base_lin_vel, base_ang_vel, dof_pos, dof_vel, self.last_action
        ], dim=-1)
        return proprio_obs

    def run_evaluation(self, save_gate_path=None):
        """执行评估并记录门控权重"""
        print(f"\n🚀 Starting evaluation on {self.motion_path}")
        print(f"   Motion frames: {self.motion_len}, Control freq: {1/self.control_dt:.1f}Hz")
        
        total_steps = int(self.sim_duration / self.control_dt)
        eval_steps = min(total_steps, self.motion_len)
        
        gate_weights_all = []
        
        with torch.no_grad():
            for step in tqdm(range(eval_steps), desc="Evaluating"):
                proprio_obs = self.get_proprio_obs(step)
                future_obs = self.get_future_obs(step).unsqueeze(0)
                
                self.proprio_history_buffer.append(proprio_obs)
                policy_proprio_history = torch.stack(list(self.proprio_history_buffer), dim=0).unsqueeze(0)
                
                if step == 0:
                    print(f"[DEBUG] proprio: {policy_proprio_history.shape}, future: {future_obs.shape}")
                
                # 调用 analyzer（内部调用 _forward_actor_gate + 专家融合）
                action, gate_weights = self.analyzer(policy_proprio_history, future_obs)
                
                if step % 50 == 0:
                    gw = gate_weights.squeeze(0).cpu().numpy()
                    dominant = np.argmax(gw)
                    print(f"[DEBUG] step={step}, dominant_expert={dominant}, max_weight={gw[dominant]:.4f}")
                
                gate_weights_cpu = gate_weights.squeeze(0).cpu().numpy()
                gate_weights_all.append(gate_weights_cpu)
                
                self.last_action = action.squeeze(0).clone()
        
        gate_weights_all = np.array(gate_weights_all)
        self._print_gate_statistics(gate_weights_all)
        
        if save_gate_path:
            os.makedirs(os.path.dirname(save_gate_path), exist_ok=True)
            np.savez(save_gate_path, gate_weights=gate_weights_all, motion_name=os.path.basename(self.motion_path))
            print(f"✓ Gate weights saved to {save_gate_path}")
        
        return gate_weights_all

    def _print_gate_statistics(self, gate_weights: np.ndarray):
        """打印门控权重统计信息（显示所有专家）"""
        T, num_experts = gate_weights.shape
        print(f"\n{'='*60}\n📊 Gate Weights Statistics\n{'='*60}")
        print(f"Timesteps: {T}, Experts: {num_experts}\n" + "-"*60)
        
        # 1. 平均权重 - ✅ 移除过滤，显示全部
        mean_weights = gate_weights.mean(axis=0)
        print(f"\n📈 Average gate weights (all {num_experts} experts):")
        for i, w in enumerate(mean_weights):
            # 用不同标记区分活跃/非活跃专家
            marker = "🟢" if w > 0.05 else "🟡" if w > 0.01 else "⚪"
            print(f"   {marker} Expert {i:2d}: {w:6.4f} ({w*100:5.2f}%)")
        
        # 2. 主导专家分布
        dominant_expert = np.argmax(gate_weights, axis=1)
        unique, counts = np.unique(dominant_expert, return_counts=True)
        print(f"\n🏆 Dominant expert distribution:")
        for exp_id in range(num_experts):  # ✅ 遍历所有专家ID
            count = counts[unique == exp_id].sum()
            if count > 0:
                pct = count / T * 100
                print(f"   Expert {exp_id:2d}: {count:4d}/{T} ({pct:5.2f}%)")
        
        # 3. 熵分析
        eps = 1e-8
        entropy = -np.sum(gate_weights * np.log(gate_weights + eps), axis=1)
        print(f"\n🎯 Gate entropy (lower = more decisive):")
        print(f"   Mean: {entropy.mean():.4f}, Std: {entropy.std():.4f}")
        
        # 4. 专家激活频率（显示全部）
        threshold = 0.1
        print(f"\n⚡ Expert activation frequency (weight > {threshold}):")
        for i in range(num_experts):  # ✅ 遍历所有专家
            cnt = (gate_weights[:, i] > threshold).sum()
            pct = cnt / T * 100
            marker = "🟢" if pct > 50 else "🟡" if pct > 10 else "⚪"
            print(f"   {marker} Expert {i:2d}: {cnt:4d}/{T} ({pct:5.2f}%)")
        
        print(f"{'='*60}\n")

# ========== 主函数 ==========
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MOETransformer Gate Weights Analyzer")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to PyTorch checkpoint')
    parser.add_argument('--motion', type=str, required=True, help='Path to motion file')
    parser.add_argument('--robot', type=str, default='g1', choices=['g1'])
    parser.add_argument('--device', type=str, default='cpu', help='cuda or cpu')
    parser.add_argument('--save-gate', type=str, default=None, help='Save gate weights to .npz')
    args = parser.parse_args()
    
    assert os.path.exists(args.checkpoint), f"Checkpoint not found: {args.checkpoint}"
    assert os.path.exists(args.motion), f"Motion not found: {args.motion}"
    
    device = args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    evaluator = MOETransformerEvaluator(
        checkpoint_path=args.checkpoint,
        motion_path=args.motion,
        robot_type=args.robot,
        device=device
    )
    evaluator.run_evaluation(save_gate_path=args.save_gate)