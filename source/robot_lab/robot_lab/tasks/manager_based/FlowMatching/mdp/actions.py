from __future__ import annotations
import torch
from typing import TYPE_CHECKING
from isaaclab.envs.mdp.actions import JointPositionAction, JointAction, JointActionCfg, JointPositionActionCfg
from isaaclab.utils import configclass
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from . import actions_cfg

class ResidualJointPositionAction(JointAction):
    cfg: actions_cfg.ResidualJointPositionActionCfg

    def __init__(self, cfg: actions_cfg.ResidualJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.command_name = cfg.command_name

    def apply_actions(self):
        command: MotionCommand = self._env.command_manager.get_term(self.command_name)
        joint_pos = command.joint_pos 
        current_actions = self.processed_actions + joint_pos[:, self._joint_ids]
        self._asset.set_joint_position_target(current_actions, joint_ids=self._joint_ids)

@configclass
class ResidualJointPositionActionCfg(JointActionCfg):
    class_type: type[ActionTerm] = ResidualJointPositionAction
    command_name: str = "motion"