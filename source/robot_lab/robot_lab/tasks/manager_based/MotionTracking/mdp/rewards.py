# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_error_magnitude

from robot_lab.tasks.manager_based.beyondmimic.mdp.commands import MotionCommand
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

#-----------------------------anchor----------------------------
def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)

def motion_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)

def motion_global_anchor_linear_velocity_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_lin_vel_w - command.robot_anchor_lin_vel_w), dim=-1)
    return torch.exp(-error / std**2)

def motion_global_body_angular_velocity_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.body_ang_vel_w - command.robot_body_ang_vel_w), dim=-1)
    return torch.exp(-error.mean(-1) / std**2)

#-----------------------------body global----------------------------
def motion_global_body_position_error_exp(env :ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.body_pos_w - command.robot_body_pos_w), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)

def motion_global_body_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.body_quat_w, command.robot_body_quat_w) ** 2
    return torch.exp(-error.mean(-1) / std**2)

def motion_global_body_linear_velocity_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.body_lin_vel_w - command.robot_body_lin_vel_w), dim=-1)
    return torch.exp(-error.mean(-1) / std**2)

def motion_global_body_angular_velocity_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.body_ang_vel_w - command.robot_body_ang_vel_w), dim=-1)
    return torch.exp(-error.mean(-1) / std**2)

#-----------------------------body relative global----------------------------
def motion_relative_body_position_error_exp(env :ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.body_pos_relative_w - command.robot_body_pos_w), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)

def motion_relative_body_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.body_quat_relative_w, command.robot_body_quat_w) ** 2
    return torch.exp(-error.mean(-1) / std**2)

def motion_relative_body_lin_vel_error_exp(env :ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.body_lin_vel_relative_w - command.robot_body_lin_vel_w), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)

def motion_relative_body_ang_vel_exp(env :ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.body_ang_vel_relative_w - command.robot_body_ang_vel_w), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)

#-----------------------------joint----------------------------
def motion_joint_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.joint_pos - command.robot_joint_pos), dim=-1)
    return torch.exp(-error / std**2)

def motion_joint_velocity_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.joint_vel - command.robot_joint_vel), dim=-1)
    return torch.exp(-error / std**2)
#-----------------------------others----------------------------
def feet_contact_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
    return reward

def robot_orientation_balance(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, std: float) -> torch.Tensor:
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    command: MotionCommand = env.command_manager.get_term(command_name)
    robot_projected_gravity_b = math_utils.quat_apply_inverse(command.robot_anchor_quat_w, asset.data.GRAVITY_VEC_W)
    error = torch.sum(torch.square(robot_projected_gravity_b[:, :2]), dim=-1)
    return torch.exp(-error / std**2)