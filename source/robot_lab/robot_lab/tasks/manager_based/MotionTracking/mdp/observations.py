# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import (
    matrix_from_quat, 
    quat_conjugate,
    quat_apply,
    quat_apply_inverse,
    quat_mul,
    euler_xyz_from_quat,
    quat_from_euler_xyz,
    subtract_frame_transforms
)

from robot_lab.tasks.manager_based.MotionTracking.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# ----------------------------gain robot obs----------------------------------------
def get_robot_heading_inv(anchor_quat_w: torch.Tensor, num_bodies: int) -> torch.Tensor:
    roll, pitch, yaw = euler_xyz_from_quat(anchor_quat_w)
    zeros = torch.zeros_like(yaw)
    heading_quat = quat_from_euler_xyz(zeros, zeros, yaw)
    heading_inv = quat_conjugate(heading_quat)
    return heading_inv.unsqueeze(1).repeat(1, num_bodies, 1)

# current global robot root position [num_envs, 3]
def robot_anchor_pos_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.robot_anchor_pos_w.view(env.num_envs, -1)

# current global 6D robot root orientation [num_envs, 6]
def robot_anchor_ori_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    mat = matrix_from_quat(command.robot_anchor_quat_w)
    return mat[..., :2].reshape(mat.shape[0], -1)

# current global robot root linear velocity [num_envs, 3]
def robot_anchor_lin_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_lin_vel_w.view(env.num_envs, -1)

# current global robot root angular velocity [num_envs, 3]
def robot_anchor_ang_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_ang_vel_w.view(env.num_envs, -1)

# current global robot joint positions [num_envs, num_joints]
def robot_joint_pos(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_joint_pos.view(env.num_envs, -1)

# current global robot joint velocities [num_envs, num_joints]
def robot_joint_vel(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_joint_vel.view(env.num_envs, -1)

# current relative robot every link positions  [num_envs, num_bodies*3]
def robot_body_pos_r(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )

    return pos_b.view(env.num_envs, -1)

# current relative robot every link orientation [num_envs, num_bodies*6]
def robot_body_ori_r(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)

def robot_body_lin_vel_r(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    num_bodies = len(command.cfg.body_names)
    anchor_quat_inv = quat_conjugate(command.robot_anchor_quat_w)[:, None, :].repeat(1, num_bodies, 1)
    body_lin_vel_r = quat_apply(anchor_quat_inv, command.robot_body_lin_vel_w)
    return body_lin_vel_r.view(env.num_envs, -1)

def robot_body_ang_vel_r(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    num_bodies = len(command.cfg.body_names)
    anchor_quat_inv = quat_conjugate(command.robot_anchor_quat_w)[:, None, :].repeat(1, num_bodies, 1)
    body_ang_vel_r = quat_apply(anchor_quat_inv, command.robot_body_ang_vel_w)
    return body_ang_vel_r.view(env.num_envs, -1)

def robot_body_pos_yaw_r(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    num_bodies = len(command.cfg.body_names)
    anchor_pos_w = command.robot_anchor_pos_w  # [num_envs, 3]
    body_pos_w = command.robot_body_pos_w      # [num_envs, num_bodies, 3]
    root_heading_inv = get_robot_heading_inv(command.robot_anchor_quat_w, num_bodies)
    pos_diff = body_pos_w - anchor_pos_w.unsqueeze(1).repeat(1, num_bodies, 1)
    body_pos_r = quat_apply(root_heading_inv, pos_diff)
    return body_pos_r.view(env.num_envs, -1)

def robot_body_ori_yaw_r(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    num_bodies = len(command.cfg.body_names)
    body_quat_w = command.robot_body_quat_w    # [num_envs, num_bodies, 4]
    root_heading_inv = get_robot_heading_inv(command.robot_anchor_quat_w, num_bodies)
    body_quat_r = quat_mul(root_heading_inv, body_quat_w)
    mat = matrix_from_quat(body_quat_r)
    return mat[..., :2].reshape(mat.shape[0], -1)

def robot_body_lin_vel_yaw_r(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    num_bodies = len(command.cfg.body_names)
    body_lin_vel_w = command.robot_body_lin_vel_w # [num_envs, num_bodies, 3]
    root_heading_inv = get_robot_heading_inv(command.robot_anchor_quat_w, num_bodies)
    body_lin_vel_r = quat_apply(root_heading_inv, body_lin_vel_w)
    return body_lin_vel_r.view(env.num_envs, -1)

def robot_body_ang_vel_yaw_r(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    num_bodies = len(command.cfg.body_names)
    body_ang_vel_w = command.robot_body_ang_vel_w # [num_envs, num_bodies, 3]
    root_heading_inv = get_robot_heading_inv(command.robot_anchor_quat_w, num_bodies)
    body_ang_vel_r = quat_apply(root_heading_inv, body_ang_vel_w)
    return body_ang_vel_r.view(env.num_envs, -1)

# ----------------------------gain global motion obs----------------------------------------
def motion_future_frames(env: ManagerBasedEnv, command_name: str, future_steps: int) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    current_frame = command.current_frame
    motion_start = command.motion.motion_start[command.env_motion_idx]
    motion_lengths = command.motion.motion_frames[command.env_motion_idx]
    offsets = torch.arange(0, future_steps, device=env.device)
    future_frames = torch.minimum(current_frame.unsqueeze(-1) + offsets.unsqueeze(0), motion_start.unsqueeze(-1) + motion_lengths.unsqueeze(-1) - 1)
    return future_frames    # [num_envs, future_steps]

# global motion every link positions  [num_envs, future_steps, num_bodies*3]
def motion_body_pos_w(env: ManagerBasedEnv, command_name: str, future_steps: int) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    future_frames = motion_future_frames(env, command_name, future_steps)
    body_index = command.body_indexes
    motion_body_pos_w = command.motion.motion_body_pos_w[:, body_index][future_frames] + command._env.scene.env_origins[:, None, None, :]
    return motion_body_pos_w.view(env.num_envs, future_steps, -1)

# global motion every link orientations(quat)  [num_envs, future_steps, num_bodies*4]
def motion_body_quat_w(env: ManagerBasedEnv, command_name: str, future_steps: int) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    future_frames = motion_future_frames(env, command_name, future_steps)
    body_index = command.body_indexes
    motion_body_quat_w = command.motion.motion_body_quat_w[:, body_index][future_frames]
    return motion_body_quat_w.view(env.num_envs, future_steps, -1)

# global motion every link orientations(6D)  [num_envs, future_steps, num_bodies*6]
def motion_body_ori_w(env: ManagerBasedEnv, command_name: str, future_steps: int) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    future_frames = motion_future_frames(env, command_name, future_steps)
    body_index = command.body_indexes
    motion_body_quat_w = command.motion.motion_body_quat_w[:, body_index][future_frames]
    mat = matrix_from_quat(motion_body_quat_w.view(-1, 4))[..., :2]
    return mat.reshape(motion_body_quat_w.shape[0], future_steps, -1)

# global motion every link linear velocities  [num_envs, future_steps, num_bodies*3]
def motion_body_lin_vel_w(env: ManagerBasedEnv, command_name: str, future_steps: int) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    future_frames = motion_future_frames(env, command_name, future_steps)
    body_index = command.body_indexes
    motion_body_lin_vel_w = command.motion.motion_body_lin_vel_w[:, body_index][future_frames]
    return motion_body_lin_vel_w.view(env.num_envs, future_steps, -1)

# global motion every link angular velocities  [num_envs, future_steps, num_bodies*3]
def motion_body_ang_vel_w(env: ManagerBasedEnv, command_name: str, future_steps: int) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    future_frames = motion_future_frames(env, command_name, future_steps)
    body_index = command.body_indexes
    motion_body_ang_vel_w = command.motion.motion_body_ang_vel_w[:, body_index][future_frames]
    return motion_body_ang_vel_w.view(env.num_envs, future_steps, -1)

# global motion root position [num_envs, future_steps, 3]
def motion_anchor_pos_w(env: ManagerBasedEnv, command_name: str, future_steps: int) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    future_frames = motion_future_frames(env, command_name, future_steps)
    body_index = command.body_indexes
    anchor_index = command.motion_anchor_body_index
    motion_anchor_pos_w = command.motion.motion_body_pos_w[:, body_index][:, anchor_index][future_frames] + command._env.scene.env_origins[:, None, None, :]
    return motion_anchor_pos_w.view(env.num_envs, future_steps, -1)

# global motion root orientation [num_envs, future_steps, 6]
def motion_anchor_ori_w(env: ManagerBasedEnv, command_name: str, future_steps: int) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    future_frames = motion_future_frames(env, command_name, future_steps)
    body_index = command.body_indexes
    anchor_index = command.motion_anchor_body_index
    motion_anchor_quat_w = command.motion.motion_body_quat_w[:, body_index][:, anchor_index][future_frames]
    mat = matrix_from_quat(motion_anchor_quat_w.view(-1, 4))[..., :2]
    return mat.reshape(mat.shape[0], future_steps, -1)

# global motion root linear velocity [num_envs, future_steps, 3]
def motion_anchor_lin_vel_w(env: ManagerBasedEnv, command_name: str, future_steps: int) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    future_frames = motion_future_frames(env, command_name, future_steps)
    body_index = command.body_indexes
    anchor_index = command.motion_anchor_body_index
    motion_anchor_lin_vel_w = command.motion.motion_body_lin_vel_w[:, body_index][:, anchor_index][future_frames]
    return motion_anchor_lin_vel_w.view(env.num_envs, future_steps, -1)

# global motion root angular velocity [num_envs, future_steps, 3]
def motion_anchor_ang_vel_w(env: ManagerBasedEnv, command_name: str, future_steps: int) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    future_frames = motion_future_frames(env, command_name, future_steps)
    body_index = command.body_indexes
    anchor_index = command.motion_anchor_body_index
    motion_anchor_ang_vel_w = command.motion.motion_body_ang_vel_w[:, body_index][:, anchor_index][future_frames]
    return motion_anchor_ang_vel_w.view(env.num_envs, future_steps, -1)

# ----------------------------gain relative motion obs----------------------------------------
# global motion every link positions  [num_envs, future_steps, num_bodies*3]
def motion_body_pos_r(env: ManagerBasedEnv, command_name: str, future_steps: int) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    future_frames = motion_future_frames(env, command_name, future_steps)
    body_index = command.body_indexes
    motion_body_pos_r = command.motion.motion_body_pos_r[:, body_index][future_frames]
    return motion_body_pos_r.view(env.num_envs, future_steps, -1)

# global motion every link orientations(quat)  [num_envs, future_steps, num_bodies*4]
def motion_body_quat_r(env: ManagerBasedEnv, command_name: str, future_steps: int) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    future_frames = motion_future_frames(env, command_name, future_steps)
    body_index = command.body_indexes
    motion_body_quat_r = command.motion.motion_body_quat_r[:, body_index][future_frames]
    return motion_body_quat_r.view(env.num_envs, future_steps, -1)

# global motion every link orientations(6D)  [num_envs, future_steps, num_bodies*6]
def motion_body_ori_r(env: ManagerBasedEnv, command_name: str, future_steps: int) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    future_frames = motion_future_frames(env, command_name, future_steps)
    body_index = command.body_indexes
    motion_body_quat_r = command.motion.motion_body_quat_r[:, body_index][future_frames]
    mat = matrix_from_quat(motion_body_quat_r.view(-1, 4))[..., :2]
    return mat.reshape(motion_body_quat_r.shape[0], future_steps, -1)

# global motion every link linear velocities  [num_envs, future_steps, num_bodies*3]
def motion_body_lin_vel_r(env: ManagerBasedEnv, command_name: str, future_steps: int) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    future_frames = motion_future_frames(env, command_name, future_steps)
    body_index = command.body_indexes
    motion_body_lin_vel_r = command.motion.motion_body_lin_vel_r[:, body_index][future_frames]
    return motion_body_lin_vel_r.view(env.num_envs, future_steps, -1)

# global motion every link angular velocities  [num_envs, future_steps, num_bodies*3]
def motion_body_ang_vel_r(env: ManagerBasedEnv, command_name: str, future_steps: int) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    future_frames = motion_future_frames(env, command_name, future_steps)
    body_index = command.body_indexes
    motion_body_ang_vel_r = command.motion.motion_body_ang_vel_r[:, body_index][future_frames]
    return motion_body_ang_vel_r.view(env.num_envs, future_steps, -1)

# global motion root position [num_envs, future_steps, 3]
def motion_anchor_pos_r(env: ManagerBasedEnv, command_name: str, future_steps: int) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    future_frames = motion_future_frames(env, command_name, future_steps)
    body_index = command.body_indexes
    anchor_index = command.motion_anchor_body_index
    motion_anchor_pos_r = command.motion.motion_body_pos_r[:, body_index][:, anchor_index][future_frames]
    return motion_anchor_pos_r.view(env.num_envs, future_steps, -1)

# global motion root orientation [num_envs, future_steps, 6]
def motion_anchor_ori_r(env: ManagerBasedEnv, command_name: str, future_steps: int) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    future_frames = motion_future_frames(env, command_name, future_steps)
    body_index = command.body_indexes
    anchor_index = command.motion_anchor_body_index
    motion_anchor_quat_r = command.motion.motion_body_quat_r[:, body_index][:, anchor_index][future_frames]
    mat = matrix_from_quat(motion_anchor_quat_r.view(-1, 4))[..., :2]
    return mat.reshape(mat.shape[0], future_steps, -1)

# global motion root linear velocity [num_envs, future_steps, 3]
def motion_anchor_lin_vel_r(env: ManagerBasedEnv, command_name: str, future_steps: int) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    future_frames = motion_future_frames(env, command_name, future_steps)
    body_index = command.body_indexes
    anchor_index = command.motion_anchor_body_index
    motion_anchor_lin_vel_r = command.motion.motion_body_lin_vel_r[:, body_index][:, anchor_index][future_frames]
    return motion_anchor_lin_vel_r.view(env.num_envs, future_steps, -1)

# global motion root angular velocity [num_envs, future_steps, 3]
def motion_anchor_ang_vel_r(env: ManagerBasedEnv, command_name: str, future_steps: int) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    future_frames = motion_future_frames(env, command_name, future_steps)
    body_index = command.body_indexes
    anchor_index = command.motion_anchor_body_index
    motion_anchor_ang_vel_r = command.motion.motion_body_ang_vel_r[:, body_index][:, anchor_index][future_frames]
    return motion_anchor_ang_vel_r.view(env.num_envs, future_steps, -1)


# ----------------------------gain relative yaw to first motion obs----------------------------------------
def get_current_heading_inv(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    r, p, y = euler_xyz_from_quat(command.anchor_quat_w)
    zeros = torch.zeros_like(y)
    heading_quat = quat_from_euler_xyz(zeros, zeros, y)
    heading_inv = quat_conjugate(heading_quat)
    return heading_inv.unsqueeze(1) # [num_envs, 1, 4]

# relative motion every link positions  [num_envs, future_steps, num_bodies*3]
def motion_body_pos_yaw_rf(env: ManagerBasedEnv, command_name: str, future_steps: int) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    future_frames = motion_future_frames(env, command_name, future_steps)
    motion_body_pos_w = command.motion.motion_body_pos_w[:, command.body_indexes][future_frames]
    motion_current_anchor_pos_w = command.motion.motion_body_pos_w[:, command.body_indexes][:, command.motion_anchor_body_index][command.current_frame]
    diff_pos = motion_body_pos_w - motion_current_anchor_pos_w[:, None, None, :]
    heading_inv = get_current_heading_inv(env, command_name)
    B, T, N, _ = diff_pos.shape
    heading_inv = heading_inv[:, :, None, :].expand(B, T, N, 4).reshape(-1, 4)
    diff_pos = diff_pos.reshape(-1, 3)
    motion_body_pos_r = quat_apply(heading_inv, diff_pos)
    return motion_body_pos_r.view(B, T, -1)

# relative motion every link orientations(quat)  [num_envs, future_steps, num_bodies*4]
def motion_body_quat_yaw_rf(env: ManagerBasedEnv, command_name: str, future_steps: int) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    future_frames = motion_future_frames(env, command_name, future_steps)
    motion_body_quat_w = command.motion.motion_body_quat_w[:, command.body_indexes][future_frames]
    heading_inv = get_current_heading_inv(env, command_name)
    B, T, N, _ = motion_body_quat_w.shape
    heading_inv = heading_inv[:, None, None, :].expand(B, T, N, 4).reshape(-1, 4)
    motion_body_quat_w = motion_body_quat_w.reshape(-1, 4)
    motion_body_quat_r = quat_mul(heading_inv, motion_body_quat_w)
    return motion_body_quat_r.view(B, T, -1)

# relative motion every link orientations(6D)  [num_envs, future_steps, num_bodies*6]
def motion_body_ori_yaw_rf(env: ManagerBasedEnv, command_name: str, future_steps: int) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    future_frames = motion_future_frames(env, command_name, future_steps)
    motion_body_quat_w = command.motion.motion_body_quat_w[:, command.body_indexes][future_frames]
    heading_inv = get_current_heading_inv(env, command_name)
    B, T, N, _ = motion_body_quat_w.shape
    heading_inv = heading_inv[:, :, None, :].expand(B, T, N, 4).reshape(-1, 4)
    motion_body_quat_w = motion_body_quat_w.reshape(-1, 4)
    motion_body_quat_r = quat_mul(heading_inv, motion_body_quat_w)
    mat = matrix_from_quat(motion_body_quat_r)[..., :2]
    return mat.reshape(B, T, -1)

# relative motion every link linear velocities  [num_envs, future_steps, num_bodies*3]
def motion_body_lin_vel_yaw_rf(env: ManagerBasedEnv, command_name: str, future_steps: int) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    future_frames = motion_future_frames(env, command_name, future_steps)
    motion_body_lin_vel_w = command.motion.motion_body_lin_vel_w[:, command.body_indexes][future_frames]
    heading_inv = get_current_heading_inv(env, command_name)
    B, T, N, _ = motion_body_lin_vel_w.shape
    heading_inv = heading_inv[:, :, None, :].expand(B, T, N, 4).reshape(-1, 4)
    motion_body_lin_vel_w = motion_body_lin_vel_w.reshape(-1, 3)
    motion_body_lin_vel_r = quat_apply(heading_inv, motion_body_lin_vel_w)
    return motion_body_lin_vel_r.view(B, T, -1)

# relative motion every link angular velocities  [num_envs, future_steps, num_bodies*3]
def motion_body_ang_vel_yaw_rf(env: ManagerBasedEnv, command_name: str, future_steps: int) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    future_frames = motion_future_frames(env, command_name, future_steps)
    motion_body_ang_vel_w = command.motion.motion_body_ang_vel_w[:, command.body_indexes][future_frames]
    heading_inv = get_current_heading_inv(env, command_name)
    B, T, N, _ = motion_body_ang_vel_w.shape
    heading_inv = heading_inv[:, :, None, :].expand(B, T, N, 4).reshape(-1, 4)
    motion_body_ang_vel_w = motion_body_ang_vel_w.reshape(-1, 3)
    motion_body_ang_vel_r = quat_apply(heading_inv, motion_body_ang_vel_w)
    return motion_body_ang_vel_r.view(B, T, -1)

# relative motion root position [num_envs, future_steps, 3]
def motion_anchor_pos_yaw_rf(env: ManagerBasedEnv, command_name: str, future_steps: int) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    future_frames = motion_future_frames(env, command_name, future_steps)
    motion_anchor_pos_w = command.motion.motion_body_pos_w[:, command.body_indexes][:, command.motion_anchor_body_index][future_frames]
    motion_current_anchor_pos_w = command.motion.motion_body_pos_w[:, command.body_indexes][:, command.motion_anchor_body_index][command.current_frame]
    diff_pos = motion_anchor_pos_w - motion_current_anchor_pos_w[:, None, :]
    heading_inv = get_current_heading_inv(env, command_name)
    B, T, _ = diff_pos.shape
    heading_inv = heading_inv.expand(B, T, 4).reshape(-1, 4)
    diff_pos = diff_pos.reshape(-1, 3)
    motion_anchor_pos_r = quat_apply(heading_inv, diff_pos)
    return motion_anchor_pos_r.view(B, T, -1)

# relative motion root orientation [num_envs, future_steps, 6]
def motion_anchor_ori_yaw_rf(env: ManagerBasedEnv, command_name: str, future_steps: int) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    future_frames = motion_future_frames(env, command_name, future_steps)
    motion_anchor_quat_w = command.motion.motion_body_quat_w[:, command.body_indexes][:, command.motion_anchor_body_index][future_frames]
    heading_inv = get_current_heading_inv(env, command_name)
    B, T, _ = motion_anchor_quat_w.shape
    heading_inv = heading_inv.expand(B, T, 4).reshape(-1, 4)
    motion_anchor_quat_w = motion_anchor_quat_w.reshape(-1, 4)
    motion_anchor_quat_r = quat_mul(heading_inv, motion_anchor_quat_w)
    mat = matrix_from_quat(motion_anchor_quat_r)[..., :2]
    return mat.reshape(B, T, -1)

def motion_anchor_ori_yaw_rf_flat(env: ManagerBasedEnv, command_name: str, future_steps: int) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    future_frames = motion_future_frames(env, command_name, future_steps)
    motion_anchor_quat_w = command.motion.motion_body_quat_w[:, command.body_indexes][:, command.motion_anchor_body_index][future_frames]
    heading_inv = get_current_heading_inv(env, command_name)
    B, T, _ = motion_anchor_quat_w.shape
    heading_inv = heading_inv.expand(B, T, 4).reshape(-1, 4)
    motion_anchor_quat_w = motion_anchor_quat_w.reshape(-1, 4)
    motion_anchor_quat_r = quat_mul(heading_inv, motion_anchor_quat_w)
    mat = matrix_from_quat(motion_anchor_quat_r)[..., :2]
    return mat.reshape(B, -1)

# relative motion root linear velocity [num_envs, future_steps, 3]
def motion_anchor_lin_vel_yaw_rf(env: ManagerBasedEnv, command_name: str, future_steps: int) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    future_frames = motion_future_frames(env, command_name, future_steps)
    motion_anchor_lin_vel_w = command.motion.motion_body_lin_vel_w[:, command.body_indexes][:, command.motion_anchor_body_index][future_frames]
    heading_inv = get_current_heading_inv(env, command_name)
    B, T, _ = motion_anchor_lin_vel_w.shape
    heading_inv = heading_inv.expand(B, T, 4).reshape(-1, 4)
    motion_anchor_lin_vel_w = motion_anchor_lin_vel_w.reshape(-1, 3)
    motion_anchor_lin_vel_r = quat_apply(heading_inv, motion_anchor_lin_vel_w)
    return motion_anchor_lin_vel_r.view(B, T, -1)

# relative motion root angular velocity [num_envs, future_steps, 3]
def motion_anchor_ang_vel_yaw_rf(env: ManagerBasedEnv, command_name: str, future_steps: int) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    future_frames = motion_future_frames(env, command_name, future_steps)
    motion_anchor_ang_vel_w = command.motion.motion_body_ang_vel_w[:, command.body_indexes][:, command.motion_anchor_body_index][future_frames]
    heading_inv = get_current_heading_inv(env, command_name)
    B, T, _ = motion_anchor_ang_vel_w.shape
    heading_inv = heading_inv.expand(B, T, 4).reshape(-1, 4)
    motion_anchor_ang_vel_w = motion_anchor_ang_vel_w.reshape(-1, 3)
    motion_anchor_ang_vel_r = quat_apply(heading_inv, motion_anchor_ang_vel_w)
    return motion_anchor_ang_vel_r.view(B, T, -1)

# ----------------------------gain relative to first motion obs----------------------------------------
def get_motion_relative_transform(command, motion_body_pos_w, motion_body_quat_w):
    body_anchor_pos_w = command.anchor_pos_w
    body_anchor_quat_w = command.anchor_quat_w
    dims = len(motion_body_pos_w.shape)
    body_anchor_pos_w = body_anchor_pos_w.view(body_anchor_pos_w.shape[0], *([1] * (dims - 2)), 3).expand(motion_body_pos_w.shape)
    body_anchor_quat_w = body_anchor_quat_w.view(body_anchor_quat_w.shape[0], *([1] * (dims - 2)), 4).expand(*motion_body_pos_w.shape[:-1], 4)
    return subtract_frame_transforms(body_anchor_pos_w, body_anchor_quat_w, motion_body_pos_w, motion_body_quat_w)

def motion_body_pos_rf(env: ManagerBasedEnv, command_name: str, future_steps: int) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    motion_body_pos_w_val = motion_body_pos_w(env, command_name, future_steps).view(env.num_envs, future_steps, -1, 3)
    motion_body_quat_w_val = motion_body_quat_w(env, command_name, future_steps).view(env.num_envs, future_steps, -1, 4)
    motion_body_pos_r, _ = get_motion_relative_transform(command, motion_body_pos_w_val, motion_body_quat_w_val)
    return motion_body_pos_r.reshape(env.num_envs, future_steps, -1)

def motion_body_quat_rf(env: ManagerBasedEnv, command_name: str, future_steps: int) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    motion_body_pos_w_val = motion_body_pos_w(env, command_name, future_steps).reshape(env.num_envs, future_steps, -1, 3)
    motion_body_quat_w_val = motion_body_quat_w(env, command_name, future_steps).reshape(env.num_envs, future_steps, -1, 4)
    _, motion_body_quat_r = get_motion_relative_transform(command, motion_body_pos_w_val, motion_body_quat_w_val)
    return motion_body_quat_r.reshape(env.num_envs, future_steps, -1)

def motion_body_ori_rf(env: ManagerBasedEnv, command_name: str, future_steps: int) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    motion_body_pos_w_val = motion_body_pos_w(env, command_name, future_steps).view(env.num_envs, future_steps, -1, 3)
    motion_body_quat_w_val = motion_body_quat_w(env, command_name, future_steps).view(env.num_envs, future_steps, -1, 4)
    _, motion_body_quat_r = get_motion_relative_transform(command, motion_body_pos_w_val, motion_body_quat_w_val)
    return matrix_from_quat(motion_body_quat_r)[..., :2].reshape(env.num_envs, future_steps, -1)

def motion_body_lin_vel_rf(env: ManagerBasedEnv, command_name: str, future_steps: int) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    motion_body_lin_vel_w_val = motion_body_lin_vel_w(env, command_name, future_steps).view(env.num_envs, future_steps, -1, 3)
    body_anchor_quat_w = command.anchor_quat_w[:, None, None, :].expand(env.num_envs, future_steps, motion_body_lin_vel_w_val.shape[2], 4)
    motion_body_lin_vel_r, _ = subtract_frame_transforms(
        torch.zeros_like(motion_body_lin_vel_w_val),
        body_anchor_quat_w,
        motion_body_lin_vel_w_val,
        torch.tensor([0.0, 0.0, 0.0, 1.0], device=env.device)
    )
    return motion_body_lin_vel_r.reshape(env.num_envs, future_steps, -1)

def motion_body_ang_vel_rf(env: ManagerBasedEnv, command_name: str, future_steps: int) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    motion_body_ang_vel_w_val = motion_body_ang_vel_w(env, command_name, future_steps).view(env.num_envs, future_steps, -1, 3)
    body_anchor_quat_w = command.anchor_quat_w[:, None, None, :].expand(env.num_envs, future_steps, motion_body_ang_vel_w_val.shape[2], 4)
    motion_body_ang_vel_r, _ = subtract_frame_transforms(
        torch.zeros_like(motion_body_ang_vel_w_val),
        body_anchor_quat_w,
        motion_body_ang_vel_w_val,
        torch.tensor([0.0, 0.0, 0.0, 1.0], device=env.device)
    )
    return motion_body_ang_vel_r.reshape(env.num_envs, future_steps, -1)

def motion_anchor_pos_rf(env: ManagerBasedEnv, command_name: str, future_steps: int) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    motion_anchor_pos_w_val = motion_anchor_pos_w(env, command_name, future_steps).view(env.num_envs, future_steps, 3)
    identity_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=env.device).expand(env.num_envs, future_steps, 4)
    motion_anchor_pos_r, _ = get_motion_relative_transform(command, motion_anchor_pos_w_val, identity_quat)
    return motion_anchor_pos_r.view(env.num_envs, future_steps, -1)

def motion_anchor_ori_rf(env: ManagerBasedEnv, command_name: str, future_steps: int) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    motion_anchor_pos_w_val = motion_anchor_pos_w(env, command_name, future_steps).view(env.num_envs, future_steps, 3)
    motion_anchor_quat_w_val = motion_anchor_quat_w(env, command_name, future_steps).view(env.num_envs, future_steps, 4)
    _, motion_anchor_quat_r = get_motion_relative_transform(command, motion_anchor_pos_w_val, motion_anchor_quat_w_val)
    return matrix_from_quat(motion_anchor_quat_r)[..., :2].reshape(env.num_envs, future_steps, -1)

def motion_anchor_lin_vel_rf(env: ManagerBasedEnv, command_name: str, future_steps: int) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    motion_anchor_lin_vel_w_val = motion_anchor_lin_vel_w(env, command_name, future_steps).view(env.num_envs, future_steps, 3)
    body_anchor_quat_w = command.anchor_quat_w[:, None, :].expand(env.num_envs, future_steps, 4)
    motion_anchor_lin_vel_r, _ = subtract_frame_transforms(
        torch.zeros_like(motion_anchor_lin_vel_w_val),
        body_anchor_quat_w,
        motion_anchor_lin_vel_w_val,
        torch.tensor([0.0, 0.0, 0.0, 1.0], device=env.device)
    )
    return motion_anchor_lin_vel_r.view(env.num_envs, future_steps, -1)

def motion_anchor_ang_vel_rf(env: ManagerBasedEnv, command_name: str, future_steps: int) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    motion_anchor_ang_vel_w_val = motion_anchor_ang_vel_w(env, command_name, future_steps).view(env.num_envs, future_steps, 3)
    body_anchor_quat_w = command.anchor_quat_w[:, None, :].expand(env.num_envs, future_steps, 4)
    motion_anchor_ang_vel_r, _ = subtract_frame_transforms(
        torch.zeros_like(motion_anchor_ang_vel_w_val),
        body_anchor_quat_w,
        motion_anchor_ang_vel_w_val,
        torch.tensor([0.0, 0.0, 0.0, 1.0], device=env.device)
    )
    return motion_anchor_ang_vel_r.view(env.num_envs, future_steps, -1)

# ----------------------------gain motion joint obs----------------------------------------
# motion joint positions [num_envs, future_steps, num_joints]
def motion_joint_pos(env: ManagerBasedEnv, command_name: str, future_steps: int) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    future_frames = motion_future_frames(env, command_name, future_steps)
    motion_joint_pos = command.motion.motion_joint_pos[future_frames]
    return motion_joint_pos.view(env.num_envs, future_steps, -1)

# motion joint velocities [num_envs, future_steps, num_joints]
def motion_joint_vel(env: ManagerBasedEnv, command_name: str, future_steps: int) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    future_frames = motion_future_frames(env, command_name, future_steps)
    motion_joint_vel = command.motion.motion_joint_vel[future_frames]
    return motion_joint_vel.view(env.num_envs, future_steps, -1)

##########################gain robot and motion diff obs##########################
def motion_anchor_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    return pos.view(env.num_envs, -1)

def motion_anchor_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)

#----------------------------gain other obs----------------------------------------
def robot_static_friction_obs(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    materials = asset.root_physx_view.get_material_properties()
    static_friction = materials[:, :, 0]
    return static_friction

def robot_dynamic_friction_obs(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    materials = asset.root_physx_view.get_material_properties()
    dynamic_friction = materials[:, :, 1]
    return dynamic_friction

def my_projected_gravity(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    robot_anchor_quat_w = command.robot_anchor_quat_w
    gravity = torch.tensor([0.0, 0.0, -1.0], device=env.device).repeat(env.num_envs, 1)
    return quat_apply_inverse(robot_anchor_quat_w, gravity)