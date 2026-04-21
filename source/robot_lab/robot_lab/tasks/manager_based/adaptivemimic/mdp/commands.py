# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import numpy as np
import os
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class MotionLoader:
    def __init__(self, motion_folder, body_indexes: Sequence[int], device: str = "cpu"):
        self.device = device
        # assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        self.motion_folders = [motion_folder] if isinstance(motion_folder, str) else list(motion_folder)
        
        self.body_indexes = body_indexes
        
        self.motion_files = []
        for folder in self.motion_folders:
            if not os.path.isdir(folder):
                raise FileNotFoundError(f"Motion folder not found: {folder}")
            for root, dirs, files in os.walk(folder):
                for f in files:
                    if f.endswith('.npz'):
                        self.motion_files.append(os.path.join(root, f))
        print(self.motion_files)
        self.num_motions = len(self.motion_files)
        self.load_motion()

    def load_motion(self):
        motion_fps = []
        motion_name = []
        motion_joint_pos = []
        motion_joint_vel = []
        motion_body_pos_w = []
        motion_body_quat_w = []
        motion_body_lin_vel_w = []
        motion_body_ang_vel_w = []
        motion_body_pos_r = []
        motion_body_quat_r = []
        motion_body_lin_vel_r = []
        motion_body_ang_vel_r = []
        motion_frames = []
        for i, file_path in enumerate(self.motion_files):
            data = np.load(file_path)
            motion_fps.append(data["fps"])
            motion_name.append(os.path.splitext(os.path.basename(file_path))[0])
            motion_joint_pos.append(torch.tensor(data["joint_pos"], dtype=torch.float32, device=self.device))
            motion_joint_vel.append(torch.tensor(data["joint_vel"], dtype=torch.float32, device=self.device))
            motion_body_pos_w.append(torch.tensor(data["body_pos_w"], dtype=torch.float32, device=self.device))
            motion_body_quat_w.append(torch.tensor(data["body_quat_w"], dtype=torch.float32, device=self.device))
            motion_body_lin_vel_w.append(torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=self.device))
            motion_body_ang_vel_w.append(torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=self.device))
            motion_body_pos_r.append(torch.tensor(data["body_pos_r"], dtype=torch.float32, device=self.device))
            motion_body_quat_r.append(torch.tensor(data["body_quat_r"], dtype=torch.float32, device=self.device))
            motion_body_lin_vel_r.append(torch.tensor(data["body_lin_vel_r"], dtype=torch.float32, device=self.device))
            motion_body_ang_vel_r.append(torch.tensor(data["body_ang_vel_r"], dtype=torch.float32, device=self.device))
            motion_frames.append(data["joint_pos"].shape[0])
        
        self.motion_name = motion_name
        self.motion_joint_pos = torch.cat(motion_joint_pos, dim=0)  # [total_frames, 23]
        self.motion_joint_vel = torch.cat(motion_joint_vel, dim=0)
        self.motion_body_pos_w = torch.cat(motion_body_pos_w, dim=0)    # [total_frames, 27, 3]
        self.motion_body_quat_w = torch.cat(motion_body_quat_w, dim=0)
        self.motion_body_lin_vel_w = torch.cat(motion_body_lin_vel_w, dim=0)
        self.motion_body_ang_vel_w = torch.cat(motion_body_ang_vel_w, dim=0)
        self.motion_body_pos_r = torch.cat(motion_body_pos_r, dim=0)    # [total_frames, 27, 3]
        self.motion_body_quat_r = torch.cat(motion_body_quat_r, dim=0)
        self.motion_body_lin_vel_r = torch.cat(motion_body_lin_vel_r, dim=0)
        self.motion_body_ang_vel_r = torch.cat(motion_body_ang_vel_r, dim=0)
        self.motion_frames = torch.tensor(motion_frames, dtype=torch.long, device=self.device) #[20, 30...]
        self.motion_start = torch.cat([torch.zeros(1, dtype=torch.long, device=self.device), self.motion_frames.cumsum(dim=0)[:-1]]) #[0, 20, 50...]


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        print("IsaacLab 中的所有 Body 名称 (self.robot.body_names):", self.robot.body_names)
        print("IsaacLab中的所有关节名称:", self.robot.data.joint_names)
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )

        self.env_motion_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.motion = MotionLoader(self.cfg.motion_folder, self.body_indexes, device=self.device)
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0
        self.body_lin_vel_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_ang_vel_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.motion_failed_count = torch.zeros(self.motion.num_motions, dtype=torch.float, device=self.device)
        self._current_motion_failed = torch.zeros(self.motion.num_motions, dtype=torch.float, device=self.device)
        # 直接每个运动序列分成的片段数量相同（舍弃每个运动序列动态片段）
        self.bin_count = self.cfg.bin_count
        self.bin_failed_count = torch.zeros(self.motion.num_motions, self.bin_count, dtype=torch.float, device=self.device)
        self._current_bin_failed = torch.zeros_like(self.bin_failed_count)
        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)], device=self.device
        )
        self.kernel = self.kernel / self.kernel.sum()

        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        # self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
        # self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        # self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:  # TODO Consider again if this is the best observation
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def current_frame(self) -> torch.Tensor:
        return self.motion.motion_start[self.env_motion_idx] + self.time_steps

    @property
    def motion_type_id(self) -> torch.Tensor:
        return self.motion.motion_type_ids[self.env_motion_idx]

    def motion_type_name(self, name: str) -> int:
        return self.motion.unique_types.index(name)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.motion_joint_pos[self.current_frame]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.motion_joint_vel[self.current_frame]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self.motion.motion_body_pos_w[self.current_frame][:, self.body_indexes] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.motion_body_quat_w[self.current_frame][:, self.body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion.motion_body_lin_vel_w[self.current_frame][:, self.body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion.motion_body_ang_vel_w[self.current_frame][:, self.body_indexes]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.motion.motion_body_pos_w[self.current_frame][:, self.body_indexes][:, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.motion.motion_body_quat_w[self.current_frame][:, self.body_indexes][:, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.motion.motion_body_lin_vel_w[self.current_frame][:, self.body_indexes][:, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.motion.motion_body_ang_vel_w[self.current_frame][:, self.body_indexes][:, self.motion_anchor_body_index]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)

        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(
            dim=-1
        )

        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(
            dim=-1
        )

        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        episode_failed = self._env.termination_manager.terminated[env_ids]
        # adaptive sampling motion
        if torch.any(episode_failed):
            fail_motion = self.env_motion_idx[env_ids][episode_failed]
            self._current_motion_failed[:] = torch.bincount(fail_motion, minlength=self.motion.num_motions)
        
        if self.cfg.enable_motion_adaptive_sampling:
            motion_sampling_probabilities = self.motion_failed_count + self.cfg.motion_adaptive_uniform_ratio / float(self.motion.num_motions)
            motion_sampling_probabilities = motion_sampling_probabilities / motion_sampling_probabilities.sum()
            sampled_motion = torch.multinomial(motion_sampling_probabilities, len(env_ids), replacement=True)
        else:
            sampled_motion =  torch.randint(0, self.motion.num_motions, (len(env_ids),), device=self.device)

        # adaptive sampling clip
        if self.cfg.enable_bin_adaptive_sampling:
            if torch.any(episode_failed):
                current_bin_index = torch.clamp(
                    (self.time_steps * self.bin_count) // self.motion.motion_frames[self.env_motion_idx], 0, self.bin_count - 1
                )       # [num_envs]
                fail_bins = current_bin_index[env_ids][episode_failed]
                self._current_bin_failed[self.env_motion_idx[env_ids][episode_failed], fail_bins] += 1.0

            self.env_motion_idx[env_ids] = sampled_motion

            bin_sampling_probabilities = self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(self.bin_count)    #[num_motions, bin_count]
            bin_sampling_probabilities = torch.nn.functional.pad(
                bin_sampling_probabilities.unsqueeze(1),
                (0, self.cfg.adaptive_kernel_size - 1),  # Non-causal kernel
                mode="replicate",
            )
            bin_sampling_probabilities = torch.nn.functional.conv1d(bin_sampling_probabilities, self.kernel.view(1, 1, -1)).squeeze(1)  # [num_motions, bin_count]

            bin_sampling_probabilities = bin_sampling_probabilities / bin_sampling_probabilities.sum(dim=1, keepdim=True)   # [num_motions, bin_count]

            sampled_bins = torch.multinomial(bin_sampling_probabilities[sampled_motion], 1, replacement=True).squeeze(-1)    # [len(env_ids)]

            motion_lengths = self.motion.motion_frames[self.env_motion_idx[env_ids]]
            start_frames = self.motion.motion_start[self.env_motion_idx[env_ids]]

            self.time_steps[env_ids] = (
                (sampled_bins.float() / self.bin_count * motion_lengths).long() + 
                (torch.rand(len(env_ids), device=self.device) * (motion_lengths / self.bin_count).long().clamp(min=1)).long()
            ).long()

            # Metrics
            H = -(bin_sampling_probabilities * (bin_sampling_probabilities + 1e-12).log()).sum()
            H_norm = H / math.log(self.bin_count)
            pmax, imax = bin_sampling_probabilities.max(dim=0)
            # self.metrics["bin_sampling_entropy"][:] = H_norm[self.env_motion_idx]
            # self.metrics["bin_sampling_top1_prob"][:] = pmax
            # self.metrics["bin_sampling_top1_bin"][:] = imax.float() / self.bin_count
        else:
            self.env_motion_idx[env_ids] = sampled_motion
            # self.time_steps[env_ids] = 0
            motion_lengths = self.motion.motion_frames[sampled_motion]
            self.time_steps[env_ids] = (torch.rand(len(env_ids), device=self.device) * motion_lengths).long()

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        self._adaptive_sampling(env_ids)

        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )
        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )

    def _update_command(self):
        self.time_steps += 1
        env_ids = torch.where(self.time_steps >= self.motion.motion_frames[self.env_motion_idx])[0]
        self._resample_command(env_ids)

        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)
        self.body_lin_vel_relative_w = quat_apply(delta_ori_w, self.body_lin_vel_w)
        self.body_ang_vel_relative_w = quat_apply(delta_ori_w, self.body_ang_vel_w)
        # update motion and bin
        self.motion_failed_count = (
            self.cfg.adaptive_alpha * self._current_motion_failed +
            (1 - self.cfg.adaptive_alpha) * self.motion_failed_count
        )
        self.bin_failed_count = (
            self.cfg.adaptive_alpha * self._current_bin_failed + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
        )
        self._print_motion_stats()
        self._current_motion_failed.zero_()
        self._current_bin_failed.zero_()

    def _print_motion_stats(self):
        print_interval = 24 * 5
        
        if self._env.common_step_counter % print_interval == 0:
            k = min(10, self.motion.num_motions)
            top_vals, top_idxs = torch.topk(self.motion_failed_count, k=k)
            print(f"\n" + "-" * 60)
            print(f"统计时刻 (Total Steps): {self._env.common_step_counter}")
            print(f"当前最难训练的 Top {k} 运动序列排行：")
            print("-" * 60)
            print(f"{'排名':<4} | {'运动序列名称':<35} | {'失败得分 (EMA)':<10}")
            print("-" * 60)
            for i in range(k):
                name = self.motion.motion_name[top_idxs[i].item()]
                score = top_vals[i].item()
                print(f"{i+1:<6} | {name:<35} | {score:<10.4f}")
            
            print("-" * 60 + "\n")

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor")
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
                )

                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name)
                        )
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name)
                        )
                    )

            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)

        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)

        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i])


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MotionCommand

    asset_name: str = MISSING

    motion_folder: str = MISSING
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)
    enable_bin_adaptive_sampling: bool = True
    enable_motion_adaptive_sampling: bool = True
    bin_count = 5

    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001
    motion_adaptive_uniform_ratio: float = 0.1

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)