# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR

# fmt: off
# Armature values per joint group
ARMATURE_LEG_PITCH = 0.06999046   # leg_pelvic_pitch, leg_knee_pitch
ARMATURE_LEG_ROLL  = 0.06999046   # leg_pelvic_roll
ARMATURE_LEG_YAW   = 0.03959369   # leg_pelvic_yaw
ARMATURE_ANKLE     = 0.03959369   # ankle_pitch, ankle_roll

ARMATURE_ARM_PITCH = 0.03298028   # shoulder_pitch, elbow_pitch
ARMATURE_ARM_ROLL  = 0.03298028   # shoulder_roll
ARMATURE_ARM_YAW   = 0.02452611   # shoulder_yaw, wrist_yaw

NATURAL_FREQ   = 10 * 2.0 * 3.1415926535  # 10 Hz
DAMPING_RATIO  = 2.0

def _stiffness(armature): return armature * NATURAL_FREQ**2
def _damping(armature):   return 2.0 * DAMPING_RATIO * armature * NATURAL_FREQ

# fmt: on

CASBOT_SKELETON_25DOF_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=False,
        asset_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/casbot_skeleton/02_description/urdf/casbot_skeleton_25dof.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.865),
        joint_pos={
            ".*_leg_pelvic_pitch_joint": -0.1,
            ".*_leg_knee_pitch_joint":    0.5,
            ".*_leg_ankle_pitch_joint":  -0.175,
            ".*_leg_ankle_roll_joint":    0.0,
            "left_shoulder_pitch_joint":  0.0,
            "right_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint":   0.0,
            "right_shoulder_roll_joint":  0.0,
            ".*_elbow_pitch_joint":       -0.5,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_leg_pelvic_pitch_joint",
                ".*_leg_pelvic_roll_joint",
                ".*_leg_pelvic_yaw_joint",
                ".*_leg_knee_pitch_joint",
            ],
            effort_limit_sim={
                ".*_leg_pelvic_pitch_joint": 150.0,
                ".*_leg_pelvic_roll_joint":  150.0,
                ".*_leg_pelvic_yaw_joint":    60.0,
                ".*_leg_knee_pitch_joint":   150.0,
            },
            velocity_limit_sim=14.0,
            stiffness={
                ".*_leg_pelvic_pitch_joint": _stiffness(ARMATURE_LEG_PITCH),
                ".*_leg_pelvic_roll_joint":  _stiffness(ARMATURE_LEG_ROLL),
                ".*_leg_pelvic_yaw_joint":   _stiffness(ARMATURE_LEG_YAW),
                ".*_leg_knee_pitch_joint":   _stiffness(ARMATURE_LEG_PITCH),
            },
            damping={
                ".*_leg_pelvic_pitch_joint": _damping(ARMATURE_LEG_PITCH),
                ".*_leg_pelvic_roll_joint":  _damping(ARMATURE_LEG_ROLL),
                ".*_leg_pelvic_yaw_joint":   _damping(ARMATURE_LEG_YAW),
                ".*_leg_knee_pitch_joint":   _damping(ARMATURE_LEG_PITCH),
            },
            armature={
                ".*_leg_pelvic_pitch_joint": ARMATURE_LEG_PITCH,
                ".*_leg_pelvic_roll_joint":  ARMATURE_LEG_ROLL,
                ".*_leg_pelvic_yaw_joint":   ARMATURE_LEG_YAW,
                ".*_leg_knee_pitch_joint":   ARMATURE_LEG_PITCH,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_leg_ankle_pitch_joint",
                ".*_leg_ankle_roll_joint",
            ],
            effort_limit_sim=60.0,
            velocity_limit_sim=14.0,
            stiffness=_stiffness(ARMATURE_ANKLE),
            damping=_damping(ARMATURE_ANKLE),
            armature=ARMATURE_ANKLE,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist_yaw_joint"],
            effort_limit_sim=60.0,
            velocity_limit_sim=14.0,
            stiffness=_stiffness(ARMATURE_LEG_YAW),
            damping=_damping(ARMATURE_LEG_YAW),
            armature=ARMATURE_LEG_YAW,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_pitch_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 75.0,
                ".*_shoulder_roll_joint":  75.0,
                ".*_shoulder_yaw_joint":   36.0,
                ".*_elbow_pitch_joint":    75.0,
                ".*_wrist_yaw_joint":      36.0,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 12.2,
                ".*_shoulder_roll_joint":  12.2,
                ".*_shoulder_yaw_joint":    9.3,
                ".*_elbow_pitch_joint":    12.2,
                ".*_wrist_yaw_joint":       9.3,
            },
            stiffness={
                ".*_shoulder_pitch_joint": _stiffness(ARMATURE_ARM_PITCH),
                ".*_shoulder_roll_joint":  _stiffness(ARMATURE_ARM_ROLL),
                ".*_shoulder_yaw_joint":   _stiffness(ARMATURE_ARM_YAW),
                ".*_elbow_pitch_joint":    _stiffness(ARMATURE_ARM_PITCH),
                ".*_wrist_yaw_joint":      _stiffness(ARMATURE_ARM_YAW),
            },
            damping={
                ".*_shoulder_pitch_joint": _damping(ARMATURE_ARM_PITCH),
                ".*_shoulder_roll_joint":  _damping(ARMATURE_ARM_ROLL),
                ".*_shoulder_yaw_joint":   _damping(ARMATURE_ARM_YAW),
                ".*_elbow_pitch_joint":    _damping(ARMATURE_ARM_PITCH),
                ".*_wrist_yaw_joint":      _damping(ARMATURE_ARM_YAW),
            },
            armature={
                ".*_shoulder_pitch_joint": ARMATURE_ARM_PITCH,
                ".*_shoulder_roll_joint":  ARMATURE_ARM_ROLL,
                ".*_shoulder_yaw_joint":   ARMATURE_ARM_YAW,
                ".*_elbow_pitch_joint":    ARMATURE_ARM_PITCH,
                ".*_wrist_yaw_joint":      ARMATURE_ARM_YAW,
            },
        ),
    },
)

CASBOT_SKELETON_25DOF_ACTION_SCALE = {}
for _a in CASBOT_SKELETON_25DOF_CFG.actuators.values():
    _e = _a.effort_limit_sim
    _s = _a.stiffness
    _names = _a.joint_names_expr
    if not isinstance(_e, dict):
        _e = {n: _e for n in _names}
    if not isinstance(_s, dict):
        _s = {n: _s for n in _names}
    for _n in _names:
        if _n in _e and _n in _s and _s[_n]:
            CASBOT_SKELETON_25DOF_ACTION_SCALE[_n] = 0.25 * _e[_n] / _s[_n]
