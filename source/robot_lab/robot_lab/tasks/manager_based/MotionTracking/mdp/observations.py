# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms

from robot_lab.tasks.manager_based.beyondmimic.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# ----------------------------gain robot obs----------------------------------------
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
def robot_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
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
def robot_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
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

# ----------------------------gain motion obs----------------------------------------
def motion_future_frames(env: ManagerBasedEnv, command_name: str, future_steps: int) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    current_frame = command.current_frame
    motion_start = command.motion.motion_start[command.env_motion_idx]
    motion_lengths = command.motion.motion_frames[command.env_motion_idx]
    offsets = torch.arange(future_steps, device=env.device)
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

# relative motion every link positions  [num_envs, future_steps, num_bodies*3]
def motion_body_pos_r(env: ManagerBasedEnv, command_name: str, future_steps: int) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    future_frames = motion_future_frames(env, command_name, future_steps)
    body_index = command.body_indexes
    motion_body_pos_r = command.motion.motion_body_pos_r[:, body_index][future_frames]
    return motion_body_pos_r.view(env.num_envs, future_steps, -1)

# relative motion every link orientations(quat)  [num_envs, future_steps, num_bodies*4]
def motion_body_quat_r(env: ManagerBasedEnv, command_name: str, future_steps: int) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    future_frames = motion_future_frames(env, command_name, future_steps)
    body_index = command.body_indexes
    motion_body_quat_r = command.motion.motion_body_quat_r[:, body_index][future_frames]
    return motion_body_quat_r.view(env.num_envs, future_steps, -1)

# relative motion every link orientations(6D)  [num_envs, future_steps, num_bodies*6]
def motion_body_ori_r(env: ManagerBasedEnv, command_name: str, future_steps: int) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    future_frames = motion_future_frames(env, command_name, future_steps)
    body_index = command.body_indexes
    motion_body_quat_r = command.motion.motion_body_quat_r[:, body_index][future_frames]
    mat = matrix_from_quat(motion_body_quat_r.view(-1, 4))[..., :2]
    return mat.reshape(motion_body_quat_r.shape[0], future_steps, -1)

# relative motion every link linear velocities  [num_envs, future_steps, num_bodies*3]
def motion_body_lin_vel_r(env: ManagerBasedEnv, command_name: str, future_steps: int) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    future_frames = motion_future_frames(env, command_name, future_steps)
    body_index = command.body_indexes
    motion_body_lin_vel_r = command.motion.motion_body_lin_vel_r[:, body_index][future_frames]
    return motion_body_lin_vel_r.view(env.num_envs, future_steps, -1)

# relative motion every link angular velocities  [num_envs, future_steps, num_bodies*3]
def motion_body_ang_vel_r(env: ManagerBasedEnv, command_name: str, future_steps: int) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    future_frames = motion_future_frames(env, command_name, future_steps)
    body_index = command.body_indexes
    motion_body_ang_vel_r = command.motion.motion_body_ang_vel_r[:, body_index][future_frames]
    return motion_body_ang_vel_r.view(env.num_envs, future_steps, -1)

# relative motion root position [num_envs, future_steps, 3]
def motion_anchor_pos_r(env: ManagerBasedEnv, command_name: str, future_steps: int) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    future_frames = motion_future_frames(env, command_name, future_steps)
    body_index = command.body_indexes
    anchor_index = command.motion_anchor_body_index
    motion_anchor_pos_r = command.motion.motion_body_pos_r[:, body_index][:, anchor_index][future_frames]
    return motion_anchor_pos_r.view(env.num_envs, future_steps, -1)

# relative motion root orientation [num_envs, future_steps, 6]
def motion_anchor_ori_r(env: ManagerBasedEnv, command_name: str, future_steps: int) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    future_frames = motion_future_frames(env, command_name, future_steps)
    body_index = command.body_indexes
    anchor_index = command.motion_anchor_body_index
    motion_anchor_quat_r = command.motion.motion_body_quat_r[:, body_index][:, anchor_index][future_frames]
    mat = matrix_from_quat(motion_anchor_quat_r.view(-1, 4))[..., :2]
    return mat.reshape(mat.shape[0], future_steps, -1)

# relative motion root linear velocity [num_envs, future_steps, 3]
def motion_anchor_lin_vel_r(env: ManagerBasedEnv, command_name: str, future_steps: int) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    future_frames = motion_future_frames(env, command_name, future_steps)
    body_index = command.body_indexes
    anchor_index = command.motion_anchor_body_index
    motion_anchor_lin_vel_r = command.motion.motion_body_lin_vel_r[:, body_index][:, anchor_index][future_frames]
    return motion_anchor_lin_vel_r.view(env.num_envs, future_steps, -1)

# relative motion root angular velocity [num_envs, future_steps, 3]
def motion_anchor_ang_vel_r(env: ManagerBasedEnv, command_name: str, future_steps: int) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    future_frames = motion_future_frames(env, command_name, future_steps)
    body_index = command.body_indexes
    anchor_index = command.motion_anchor_body_index
    motion_anchor_ang_vel_r = command.motion.motion_body_ang_vel_r[:, body_index][:, anchor_index][future_frames]
    return motion_anchor_ang_vel_r.view(env.num_envs, future_steps, -1)

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
