import argparse
import os
import numpy as np
import mujoco
import mujoco_viewer
from tqdm import tqdm
import torch
from tensordict import TensorDict

import utils.math_utils as math_utils

from rsl_rl.modules import ActorCritic, StudentTeacher

class HumanoidEnv:
    def __init__(self, policy_path, motion_path, robot_type="g1", device="cpu", record_video=False):
        self.robot_type = robot_type
        self.device = device
        self.record_video = record_video
        self.motion_path = motion_path
        self.policy_path = policy_path

        self.load_motion()

        self.anchor_index = 4
        self.anchor_name = "torso_link"

        self.mujoco2isaac_dof_index = torch.tensor([0, 6, 12, 1, 7, 13, 18, 2, 8, 14, 19, 3, 9, 15, 20, 4, 10, 16, 21, 5, 11, 17, 22], device=self.device)
        self.isaac2mujoco_dof_index = torch.tensor([0, 3, 7, 11, 15, 19, 1, 4, 8, 12, 16, 20, 2, 5, 9, 13, 17, 21, 6, 10, 14, 18, 22], device=self.device)

        self.motion_body_indexes = torch.tensor([0, 5, 15, 23, 6, 16, 24, 4, 13, 21, 25, 14, 22, 26], device=self.device)
        self.robot_body_indexes = torch.tensor([1, 3, 5, 7, 9, 11, 13, 14, 16, 18, 19, 21, 23, 24], device=self.device)

        if robot_type == "g1":
            model_path = "/home/ubuntu/Desktop/hjq/assets/g1_23dof_rev_1_0.xml"
            self.stiffness = np.array([ 
                40.179, 99.098, 40.179, 99.098, 28.501, 28.501,
                40.179, 99.098, 40.179, 99.098, 28.501, 28.501,
                40.179,
                14.25, 14.25, 14.25, 14.25, 14.25,
                14.25, 14.25, 14.25, 14.25, 14.25
            ])
            self.damping = np.array([ 
                2.557, 6.308, 2.557, 6.308, 1.814, 1.814,
                2.557, 6.308, 2.557, 6.308, 1.814, 1.814,
                2.557,
                0.907, 0.907, 0.907, 0.907, 0.907,
                0.907, 0.907, 0.907, 0.907, 0.907
            ])
            self.torque_limits = np.array([ 
                88.0, 139.0, 88.0, 139.0, 35.0, 35.0,
                88.0, 139.0, 88.0, 139.0, 35.0, 35.0,
                88.0,
                25.0, 25.0, 25.0, 25.0, 25.0,
                25.0, 25.0, 25.0, 25.0, 25.0
            ])

            self.default_dof_pos = torch.tensor([
                -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
                -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
                0.0,
                0.2, 0.2, 0.0, 0.6, 0.0,
                0.2, -0.2, 0.0, 0.6, 0.0
            ], device=self.device, dtype=torch.float32)

            self.action_scale = torch.tensor([
                0.547, 0.350, 0.547, 0.350, 0.307, 0.307,
                0.547, 0.350, 0.547, 0.350, 0.307, 0.307,
                0.547,
                0.438, 0.438, 0.438, 0.438, 0.438,
                0.438, 0.438, 0.438, 0.438, 0.438
            ], device=self.device, dtype=torch.float32)

            self.num_actions = 23
            self.num_dofs = 23
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
        robot_pos = torch.from_numpy(self.data.body('torso_link').xpos.astype(np.float32)).to(self.device).unsqueeze(0)
        robot_quat = torch.from_numpy(self.data.body('torso_link').xquat.astype(np.float32)).to(self.device).unsqueeze(0)

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
        state_dict = checkpoint.get('model_state_dict', checkpoint)

        self.actor_obs_dim = 124
        self.teacher_obs_dim = 130

        dummy_obs = TensorDict({
            "policy_obs": torch.zeros(1, self.actor_obs_dim, device=self.device),
            "teacher_obs": torch.zeros(1, self.teacher_obs_dim, device=self.device)
        }, batch_size=[1])

        obs_groups = {
            "policy": ["policy_obs"],
            "teacher": ["teacher_obs"]
        }

        self.policy_net = StudentTeacher(
            obs=dummy_obs,
            obs_groups=obs_groups,
            num_actions=self.num_actions,
            student_hidden_dims=[512, 256, 128],
            teacher_hidden_dims=[512, 256, 128],
            activation="elu",
            init_noise_std=0.2,
            student_obs_normalization=False,   # 根据你的训练设置调整
            teacher_obs_normalization=False,
        )
        self.policy_net.load_state_dict(state_dict, strict=True)
        self.policy_net.to(self.device)
        self.policy_net.eval()

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
            command,              # 46
            # motion_anchor_pos_b,  # 3
            motion_anchor_ori_b,  # 6
            # base_lin_vel,         # 3
            base_ang_vel,         # 3
            joint_pos,            # 23
            joint_vel,            # 23
            self.last_action,     # 23
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
                obs_dict = TensorDict({
                    "policy_obs": obs_vec,
                    "critic_obs": obs_vec,
                }, batch_size=[1])

                with torch.no_grad():
                    action = self.policy_net.act_inference(obs_dict).squeeze()

                self.last_action = action.clone()

                scaled_action = action * self.action_scale[self.mujoco2isaac_dof_index] + self.default_dof_pos[self.mujoco2isaac_dof_index]
                pd_target = scaled_action[self.isaac2mujoco_dof_index].cpu().numpy()

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

    checkpoint = "/home/ubuntu/Desktop/robot_lab/logs/rsl_rl/unitree_g1_FlowMatching_flat/2026-03-23_18-42-56/model_7000.pt"
    motion_file = "/home/ubuntu/Desktop/robot_lab/source/robot_lab/robot_lab/tasks/manager_based/FlowMatching/config/g1/motion/113_08_poses.npz"

    env = HumanoidEnv(
        policy_path=checkpoint,
        motion_path=motion_file,
        robot_type=args.robot,
        device="cpu",
        record_video=args.record_video
    )
    env.run()