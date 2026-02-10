# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import os

from isaaclab.utils import configclass

from robot_lab.assets.unitree import UNITREE_G1_29DOF_ACTION_SCALE, UNITREE_G1_29DOF_CFG, UNITREE_G1_23DOF_ACTION_SCALE, UNITREE_G1_23DOF_CFG
from robot_lab.tasks.manager_based.MotionTracking.tracking_env_cfg import MotionTrackingEnvCfg


@configclass
class UnitreeG1MotionTrackingFlatEnvCfg(MotionTrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = UNITREE_G1_23DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = UNITREE_G1_23DOF_ACTION_SCALE
        motion_base_path = f"{os.path.dirname(__file__)}/motion"
        self.commands.motion.motion_folder = [
            f"{motion_base_path}/walk",
            f"{motion_base_path}/up_down",
            f"{motion_base_path}/jump",
            f"{motion_base_path}/run",
            f"{motion_base_path}/single_stance",
            f"{motion_base_path}/dance",
        ]
        self.commands.motion.anchor_body_name = "torso_link"
        self.commands.motion.body_names = [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_roll_rubber_hand",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_roll_rubber_hand",
        ]

        self.episode_length_s = 30.0
