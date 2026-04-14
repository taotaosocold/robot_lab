import argparse
import os
import numpy as np
import mujoco
import mujoco_viewer
from tqdm import tqdm
import torch
from tensordict import TensorDict

import utils.math_utils as math_utils

from rsl_rl.models import MLPModel

class HumanoidEnv:
    def __init__(self, policy_path, motion_path, robot_type="g1", device="cpu", record_video=False):
        self.robot_type = robot_type
        self.device = device
        self.record_video = record_video
        self.motion_path = motion_path
        self.policy_path = policy_path

        self.load_motion()

        self.anchor_index = 1
        self.anchor_name = "torso_link"

        self.mujoco2isaac_dof_index = torch.tensor([0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22,4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26,20, 27, 21, 28,], device=self.device)
        self.isaac2mujoco_dof_index = torch.tensor([0, 3, 6, 9, 13, 17,1, 4, 7, 10, 14, 18,2, 5, 8, 11, 15, 19, 21, 23, 25, 27,12, 16, 20, 22, 24, 26, 28,], device=self.device)

        self.motion_body_indexes = torch.tensor([1, 5, 11, 19, 6, 12, 20, 10, 17, 23, 29, 18, 24, 30], device=self.device)
        self.robot_body_indexes = torch.tensor([1, 3, 5, 7, 9, 11, 13, 16, 18, 20, 23, 25, 27, 30], device=self.device)

        if robot_type == "g1":
            model_path = "/home/casbot/Desktop/hjq/assets/g1/g1_29dof_rev_1_0.xml"
            # stiffness = armature * (10*2π)^2, per unitree.py
            self.stiffness = np.array([
                40.179, 99.098, 40.179, 99.098, 28.501, 28.501,  # left leg: pitch, roll, yaw, knee, ankle_pitch, ankle_roll
                40.179, 99.098, 40.179, 99.098, 28.501, 28.501,  # right leg
                40.179, 28.501, 28.501,                           # waist: yaw, roll, pitch
                14.251, 14.251, 14.251, 14.251, 14.251, 16.778, 16.778,  # left arm: sh_pitch, sh_roll, sh_yaw, elbow, wr_roll, wr_pitch, wr_yaw
                14.251, 14.251, 14.251, 14.251, 14.251, 16.778, 16.778,  # right arm
            ])
            # damping = 2 * 2.0 * armature * (10*2π), per unitree.py
            self.damping = np.array([
                2.558, 6.309, 2.558, 6.309, 1.814, 1.814,
                2.558, 6.309, 2.558, 6.309, 1.814, 1.814,
                2.558, 1.814, 1.814,
                0.907, 0.907, 0.907, 0.907, 0.907, 1.068, 1.068,
                0.907, 0.907, 0.907, 0.907, 0.907, 1.068, 1.068,
            ])
            # effort_limit_sim from unitree.py
            self.torque_limits = np.array([
                88.0, 139.0, 88.0, 139.0, 50.0, 50.0,
                88.0, 139.0, 88.0, 139.0, 50.0, 50.0,
                88.0, 50.0, 50.0,
                25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0,
                25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0,
            ])

            # init_state.joint_pos from unitree.py
            self.default_dof_pos = torch.tensor([
                -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,   # left leg
                -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,   # right leg
                 0.0, 0.0, 0.0,                           # waist: yaw, roll, pitch
                 0.2,  0.2, 0.0,  0.6, 0.0, 0.0, 0.0,   # left arm: sh_pitch, sh_roll, sh_yaw, elbow, wr_roll, wr_pitch, wr_yaw
                 0.2, -0.2, 0.0,  0.6, 0.0, 0.0, 0.0,   # right arm
            ], device=self.device, dtype=torch.float32)

            # action_scale = 0.25 * effort / stiffness, per unitree.py UNITREE_G1_29DOF_ACTION_SCALE
            self.action_scale = torch.tensor([
                0.548, 0.351, 0.548, 0.351, 0.439, 0.439,
                0.548, 0.351, 0.548, 0.351, 0.439, 0.439,
                0.548, 0.439, 0.439,
                0.439, 0.439, 0.439, 0.439, 0.439, 0.075, 0.075,
                0.439, 0.439, 0.439, 0.439, 0.439, 0.075, 0.075,
            ], device=self.device, dtype=torch.float32)

            self.action_residual_scale = torch.tensor([
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            ], device=self.device, dtype=torch.float32)

            self.num_actions = 29
            self.num_dofs = 29
        else:
            raise ValueError(f"Robot type {robot_type} not supported!")

        self.sim_duration = 60.0
        self.sim_dt = 0.005
        self.sim_decimation = 4

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = self.sim_dt
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_step(self.model, self.data)

        if self.record_video:
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data, 'offscreen')
        else:
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        self.viewer.cam.distance = 5.0

        self.last_action = torch.zeros(self.num_actions, device=self.device, dtype=torch.float32)
        self.align_motion_to_robot()
        self.load_model()

    def load_motion(self):
        data = np.load(self.motion_path, allow_pickle=True)
        self.joint_pos = torch.tensor(data['joint_pos'], device=self.device, dtype=torch.float32)
        self.joint_vel = torch.tensor(data['joint_vel'], device=self.device, dtype=torch.float32)
        self.body_pos_w = torch.tensor(data['body_pos_w'], device=self.device, dtype=torch.float32)
        self.body_quat_w = torch.tensor(data['body_quat_w'], device=self.device, dtype=torch.float32)
        self.body_lin_vel_w = torch.tensor(data['body_lin_vel_w'], device=self.device, dtype=torch.float32)
        self.body_ang_vel_w = torch.tensor(data['body_ang_vel_w'], device=self.device, dtype=torch.float32)
        self.motion_len = self.joint_pos.shape[0]

    def align_motion_to_robot(self):
        robot_pos = torch.from_numpy(self.data.body(self.anchor_name).xpos.astype(np.float32)).to(self.device).unsqueeze(0)
        robot_quat = torch.from_numpy(self.data.body(self.anchor_name).xquat.astype(np.float32)).to(self.device).unsqueeze(0)

        motion_pos_start = self.body_pos_w[0, self.anchor_index].unsqueeze(0)
        motion_quat_start = self.body_quat_w[0, self.anchor_index].unsqueeze(0)

        robot_yaw_quat = math_utils.yaw_quat(robot_quat)
        motion_yaw_quat = math_utils.yaw_quat(motion_quat_start)
        yaw_offset_quat = math_utils.quat_mul(robot_yaw_quat, math_utils.quat_inv(motion_yaw_quat))
        yaw_offset_quat_exp = yaw_offset_quat.view(1, 1, 4)
        pos_offset = robot_pos.clone()
        pos_offset[..., 2] = 0

        num_frames = self.body_pos_w.shape[0]
        num_bodies = self.body_pos_w.shape[1]
        m_pos_start_xy = motion_pos_start.clone()
        m_pos_start_xy[..., 2] = 0
        
        centered_pos = self.body_pos_w - m_pos_start_xy.view(1, 1, 3)
        self.body_pos_w = math_utils.quat_apply(
            yaw_offset_quat_exp.expand(num_frames, num_bodies, 4), 
            centered_pos.reshape(-1, 3)
        ).view(num_frames, num_bodies, 3) + pos_offset.view(1, 1, 3)

        self.body_lin_vel_w = math_utils.quat_apply(
            yaw_offset_quat_exp.expand(num_frames, num_bodies, 4),
            self.body_lin_vel_w.reshape(-1, 3)
        ).view(num_frames, num_bodies, 3)
        
        self.body_ang_vel_w = math_utils.quat_apply(
            yaw_offset_quat_exp.expand(num_frames, num_bodies, 4),
            self.body_ang_vel_w.reshape(-1, 3)
        ).view(num_frames, num_bodies, 3)

        self.body_quat_w = math_utils.quat_mul(
            yaw_offset_quat_exp.expand(num_frames, num_bodies, 4).reshape(-1, 4),
            self.body_quat_w.reshape(-1, 4)
        ).view(num_frames, num_bodies, 4)

    def load_model(self):
        print(f"Loading policy from {self.policy_path}")
        checkpoint = torch.load(self.policy_path, map_location=self.device)

        actor_obs_dim = self.get_obs(0).shape[0]
        dummy_obs = TensorDict({"policy": torch.zeros(1, actor_obs_dim, device=self.device)}, batch_size=[1])
        obs_groups = {"actor": ["policy"]}

        self.actor = MLPModel(
            obs=dummy_obs,
            obs_groups=obs_groups,
            obs_set="actor",
            output_dim=self.num_actions,
            hidden_dims=[512, 256, 128],
            activation="elu",
            obs_normalization=False,
            distribution_cfg=None,
        )
        self.actor.load_state_dict(checkpoint["actor_state_dict"], strict=False)
        self.actor.to(self.device)
        self.actor.eval()

    def get_obs(self, curr_timestep):
        dof_pos = torch.from_numpy(self.data.qpos[-self.num_dofs:].astype(np.float32)).to(self.device)
        dof_vel = torch.from_numpy(self.data.qvel[-self.num_dofs:].astype(np.float32)).to(self.device)

        anchor_pos_w = torch.from_numpy(self.data.body(self.anchor_name).xpos.astype(np.float32)).to(self.device)
        anchor_quat_w = torch.from_numpy(self.data.body(self.anchor_name).xquat.astype(np.float32)).to(self.device)

        base_lin_vel = torch.from_numpy(self.data.qvel[0:3].astype(np.float32)).to(self.device)
        base_ang_vel = torch.from_numpy(self.data.qvel[3:6].astype(np.float32)).to(self.device)

        target_idx = min(curr_timestep, self.motion_len - 1)
        command = torch.cat([self.joint_pos[target_idx], self.joint_vel[target_idx]])

        # motion anchor in base frame
        rel_pos_w = self.body_pos_w[target_idx, self.anchor_index] - anchor_pos_w
        motion_anchor_pos_b = math_utils.quat_apply_inverse(anchor_quat_w, rel_pos_w)

        rel_quat_b = math_utils.quat_mul(math_utils.quat_conjugate(anchor_quat_w),
                                         self.body_quat_w[target_idx, self.anchor_index])
        motion_anchor_ori_b = math_utils.matrix_from_quat(rel_quat_b)[..., :2].reshape(-1)

        joint_pos = (dof_pos - self.default_dof_pos)[self.mujoco2isaac_dof_index]
        joint_vel = dof_vel[self.mujoco2isaac_dof_index]

        obs_vec = torch.cat([
            command,              # 58 (29*2)
            # motion_anchor_pos_b,  # 3
            motion_anchor_ori_b,  # 6
            # base_lin_vel,         # 3
            base_ang_vel,         # 3
            joint_pos,            # 29
            joint_vel,            # 29
            self.last_action,     # 29
        ], dim=-1)

        return obs_vec

    def run(self):
        motion_name = os.path.basename(self.motion_path).split('.')[0]
        if self.record_video:
            import imageio
            os.makedirs("mujoco_videos", exist_ok=True)
            video_name = os.path.join("mujoco_videos", f"{self.robot_type}_{os.path.basename(self.policy_path).split('.')[0]}_{motion_name}.mp4")
            mp4_writer = imageio.get_writer(video_name, fps=50)

        pd_target = self.default_dof_pos.cpu().numpy()

        for i in tqdm(range(int(self.sim_duration / self.sim_dt)), desc="Running simulation..."):
            curr_timestep = i // self.sim_decimation

            if i % self.sim_decimation == 0:
                obs_vec = self.get_obs(curr_timestep).unsqueeze(0)
                obs_dict = TensorDict({"policy": obs_vec}, batch_size=[1])

                with torch.no_grad():
                    action = self.actor(obs_dict).squeeze()

                self.last_action = action.clone()

                scaled_action = action * self.action_scale[self.mujoco2isaac_dof_index] + self.default_dof_pos[self.mujoco2isaac_dof_index]
                target_idx = min(curr_timestep, self.motion_len - 1)
                # residual_scaled_action = action * self.action_residual_scale[self.mujoco2isaac_dof_index] + self.joint_pos[target_idx]
                pd_target = scaled_action[self.isaac2mujoco_dof_index].cpu().numpy()
                # pd_target = residual_scaled_action[self.isaac2mujoco_dof_index].cpu().numpy()

                self.viewer.cam.lookat = self.data.qpos.astype(np.float32)[:3]
                if self.record_video:
                    img = self.viewer.read_pixels()
                    mp4_writer.append_data(img)
                else:
                    self.viewer.render()

            # PD control
            dof_pos = self.data.qpos[-self.num_dofs:]
            dof_vel = self.data.qvel[-self.num_dofs:]
            torque = (pd_target - dof_pos) * self.stiffness - dof_vel * self.damping
            torque = np.clip(torque, -self.torque_limits, self.torque_limits)
            self.data.ctrl = torque

            mujoco.mj_step(self.model, self.data)

        self.viewer.close()
        if self.record_video:
            mp4_writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot', type=str, default="g1")
    parser.add_argument('--record_video', action='store_true')
    args = parser.parse_args()

    checkpoint = "/home/casbot/Desktop/robot_lab/logs/rsl_rl/unitree_g1_beyondmimic_flat/2026-04-09_17-23-42/model_best_ep_len.pt"
    motion_file = "/home/casbot/Desktop/robot_lab/source/robot_lab/robot_lab/tasks/manager_based/beyondmimic/config/g1/motion/G1_Take_102.bvh_60hz.npz"

    env = HumanoidEnv(
        policy_path=checkpoint,
        motion_path=motion_file,
        robot_type=args.robot,
        device="cpu",
        record_video=args.record_video
    )
    env.run()