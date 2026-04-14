# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import os

from isaaclab.utils import configclass

from robot_lab.assets.unitree import UNITREE_G1_29DOF_ACTION_SCALE, UNITREE_G1_29DOF_CFG
from robot_lab.assets.casbot import CASBOT_02_25DOF_ACTION_SCALE, CASBOT_02_25DOF_CFG
from robot_lab.assets.casbot_skeleton import CASBOT_SKELETON_25DOF_ACTION_SCALE, CASBOT_SKELETON_25DOF_CFG
from robot_lab.tasks.manager_based.adaptivemimic.tracking_env_cfg import AdaptiveMimicEnvCfg


@configclass
class Casbot02AdaptiveMimicFlatEnvCfg(AdaptiveMimicEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = CASBOT_SKELETON_25DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = CASBOT_SKELETON_25DOF_ACTION_SCALE
        self.commands.motion.motion_file = f"{os.path.dirname(__file__)}/motion/fallAndGetUp2_subject2.npz"
        # self.commands.motion.motion_file = f"{os.path.dirname(__file__)}/motion/G1_gangnam_style_V01.bvh_60hz.npz"
        self.commands.motion.anchor_body_name = "waist_yaw_link"
        self.commands.motion.body_names = [
            "base_link",
            "left_leg_pelvic_roll_link",
            "left_leg_knee_pitch_link",
            "left_leg_ankle_roll_link",
            "right_leg_pelvic_roll_link",
            "right_leg_knee_pitch_link",
            "right_leg_ankle_roll_link",
            "waist_yaw_link",
            "left_shoulder_roll_link",
            "left_elbow_pitch_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_pitch_link",
            "right_wrist_yaw_link",
        ]

        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None

        self.episode_length_s = 30.0
