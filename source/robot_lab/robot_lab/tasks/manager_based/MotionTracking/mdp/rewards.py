# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_error_magnitude

from robot_lab.tasks.manager_based.MotionTracking.mdp.commands import MotionCommand
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]

def motion_type_filter_wrapper(env, command_name: str, target_types: list[str], reward_fn, reward_params: dict = None):
    command: MotionCommand = env.command_manager.get_term(command_name)
    current_type_ids = command.motion_type_id
    combined_mask = torch.zeros_like(current_type_ids, dtype=torch.bool, device=env.device)
    for t_type in target_types:
        target_id = command.motion_type_name(t_type)
        combined_mask |= (current_type_ids == target_id)
    params = reward_params if reward_params is not None else {}
    raw_reward = reward_fn(env, **params)
    return raw_reward * combined_mask.float()

#-----------------------------anchor global----------------------------
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

def motion_global_anchor_angular_velocity_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_ang_vel_w - command.robot_anchor_ang_vel_w), dim=-1)
    return torch.exp(-error.mean(-1) / std**2)

#-----------------------------body global----------------------------
def motion_global_body_position_error_exp(env :ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(torch.square(command.body_pos_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1)
    return torch.exp(-error.mean(-1) / std**2)

def motion_global_body_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = quat_error_magnitude(command.body_quat_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes]) ** 2
    return torch.exp(-error.mean(-1) / std**2)

def motion_global_body_linear_velocity_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1)
    return torch.exp(-error.mean(-1) / std**2)

def motion_global_body_angular_velocity_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1)
    return torch.exp(-error.mean(-1) / std**2)

#-----------------------------body relative global----------------------------
def motion_relative_body_position_error_exp(env :ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1)
    return torch.exp(-error.mean(-1) / std**2)

def motion_relative_body_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes]) ** 2
    return torch.exp(-error.mean(-1) / std**2)

def motion_relative_body_linear_velocity_error_exp(env :ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(torch.square(command.body_lin_vel_relative_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1)
    return torch.exp(-error.mean(-1) / std**2)

def motion_relative_body_angular_velocity_error_exp(env :ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(torch.square(command.body_ang_vel_relative_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1)
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

def single_stance(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    in_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
    body_heights = asset.data.body_pos_w[:, sensor_cfg.body_ids, 2]
    is_ground_contact = in_contact & (body_heights < 0.05)
    num_contacts = torch.sum(is_ground_contact.float(), dim=-1)
    return (num_contacts == 1).float()