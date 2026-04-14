import argparse
import os
import numpy as np
import mujoco
import mujoco_viewer
from tqdm import tqdm
import torch
from tensordict import TensorDict

import utils.math_utils as math_utils
import matplotlib.pyplot as plt
from rsl_rl.models import MLPModel

class HumanoidEnv:
    def __init__(self, policy_path, motion_path, robot_type="casbot", device="cpu", record_video=False):
        self.robot_type = robot_type
        self.device = device
        self.record_video = record_video
        self.motion_path = motion_path
        self.policy_path = policy_path

        self.load_motion()

        self.anchor_index = 0
        self.anchor_name = "waist_yaw_link"

        self.mujoco2isaac_dof_index = torch.tensor([0, 6, 12, 1, 7, 23, 13, 18, 2, 8, 24, 14, 19, 3, 9, 15, 20, 4, 10, 16, 21, 5, 11, 17, 22], device=self.device)
        self.isaac2mujoco_dof_index = torch.tensor([0, 3, 8, 13, 17, 21, 1, 4, 9, 14, 18, 22, 2, 6, 11, 15, 19, 23, 7, 12, 16, 20, 24, 5, 10], device=self.device)

        self.motion_body_indexes = torch.tensor([0, 4, 14, 22, 5, 15, 23, 3, 12, 20, 24, 13, 21, 25], device=self.device)
        self.robot_body_indexes = torch.tensor([1, 3, 5, 7, 9, 11, 13, 14, 16, 18, 19, 21, 23, 24], device=self.device)

        if robot_type == "casbot":
            model_path = "/home/casbot/Desktop/hjq/assets/casbot/casbot02_25dof_head_at_last.xml"
            # stiffness = armature * (10*2π)^2, per casbot.py
            self.stiffness = np.array([
                276.311, 276.311, 156.310, 276.311, 156.310, 156.310,  # left leg: pitch, roll, yaw, knee, ankle_pitch, ankle_roll
                276.311, 276.311, 156.310, 276.311, 156.310, 156.310,  # right leg: pitch, roll, yaw, knee, ankle_pitch, ankle_roll
                156.310,                                                 # waist_yaw
                130.201, 130.201,  96.825, 130.201,  96.825,           # left arm: sh_pitch, sh_roll, sh_yaw, elbow, wrist_yaw
                130.201, 130.201,  96.825, 130.201,  96.825,           # right arm
                96.825,  96.825,                                        # head: yaw, pitch  (armature=0.02452611, same as arm_yaw)
            ])
            # damping = 2 * 2.0 * armature * (10*2π), per casbot.py
            self.damping = np.array([
                17.591, 17.591,  9.951, 17.591,  9.951,  9.951,
                17.591, 17.591,  9.951, 17.591,  9.951,  9.951,
                9.951,
                8.289,  8.289,  6.164,  8.289,  6.164,
                8.289,  8.289,  6.164,  8.289,  6.164,
                6.164,  6.164,
            ])
            # effort_limit_sim from casbot URDF
            self.torque_limits = np.array([
                150.0, 150.0,  60.0, 150.0,  60.0,  60.0,
                150.0, 150.0,  60.0, 150.0,  60.0,  60.0,
                60.0,
                75.0,  75.0,  36.0,  75.0,  36.0,
                75.0,  75.0,  36.0,  75.0,  36.0,
                36.0,  36.0,
            ])
            # init_state.joint_pos from casbot.py (head default = 0)
            self.default_dof_pos = torch.tensor([
                -0.1, 0.0, 0.0, 0.5, -0.175, 0.0,
                -0.1, 0.0, 0.0, 0.5, -0.175, 0.0,
                0.0,
                0.0,  0.0, 0.0, -0.5, 0.0,
                0.0,  0.0, 0.0, -0.5, 0.0,
                0.0,  0.0,
            ], device=self.device, dtype=torch.float32)

            # action_scale = 0.25 * effort / stiffness, per casbot.py CASBOT_02_25DOF_ACTION_SCALE
            self.action_scale = torch.tensor([
                0.136, 0.136, 0.096, 0.136, 0.096, 0.096,
                0.136, 0.136, 0.096, 0.136, 0.096, 0.096,
                0.096,
                0.144, 0.144, 0.093, 0.144, 0.093,
                0.144, 0.144, 0.093, 0.144, 0.093,
                0.093, 0.093,
            ], device=self.device, dtype=torch.float32)

            self.action_residual_scale = torch.tensor([
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0,
                1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0,
            ], device=self.device, dtype=torch.float32)

            self.num_actions = 25
            self.num_dofs = 25
        else:
            raise ValueError(f"Robot type {robot_type} not supported!")

        self.sim_duration = 10.0
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
            command,              # 25*2
            # motion_anchor_pos_b,  # 3
            motion_anchor_ori_b,  # 6
            # base_lin_vel,         # 3
            base_ang_vel,         # 3
            joint_pos,            # 25
            joint_vel,            # 25
            self.last_action,     # 25
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

        # ---------- 记录力矩和时间 ----------
        torque_history = []      # 每个元素为长度25的numpy数组
        time_history = []        # 仿真时间(秒)

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
                pd_target = scaled_action[self.isaac2mujoco_dof_index].cpu().numpy()

                self.viewer.cam.lookat = self.data.qpos.astype(np.float32)[:3]
                if self.record_video:
                    img = self.viewer.read_pixels()
                    mp4_writer.append_data(img)
                else:
                    self.viewer.render()

            # PD控制
            dof_pos = self.data.qpos[-self.num_dofs:]
            dof_vel = self.data.qvel[-self.num_dofs:]
            torque = (pd_target - dof_pos) * self.stiffness - dof_vel * self.damping
            torque = np.clip(torque, -self.torque_limits, self.torque_limits)
            self.data.ctrl = torque

            # 记录
            torque_history.append(torque.copy())
            time_history.append(i * self.sim_dt)

            mujoco.mj_step(self.model, self.data)

        self.viewer.close()
        if self.record_video:
            mp4_writer.close()

        # ---------- 绘制力矩曲线 ----------
        time_arr = np.array(time_history)
        torque_arr = np.array(torque_history)  # shape: (steps, 25)

        # 关节名称（与 stiffness 数组顺序一致）
        joint_names = [
            "L_hip_pitch", "L_hip_roll", "L_hip_yaw", "L_knee", "L_ankle_pitch", "L_ankle_roll",
            "R_hip_pitch", "R_hip_roll", "R_hip_yaw", "R_knee", "R_ankle_pitch", "R_ankle_roll",
            "Waist_yaw",
            "L_sh_pitch", "L_sh_roll", "L_sh_yaw", "L_elbow", "L_wrist_yaw",
            "R_sh_pitch", "R_sh_roll", "R_sh_yaw", "R_elbow", "R_wrist_yaw",
            "Head_yaw", "Head_pitch"
        ]

        # 创建三个子图：腿+腰、手臂、头部
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        fig.suptitle(f"Joint Torques for Motion: {motion_name}", fontsize=14)

        # 腿部 + 腰部（索引 0~12）
        ax1 = axes[0]
        for idx in range(13):
            ax1.plot(time_arr, torque_arr[:, idx], label=joint_names[idx], linewidth=0.8)
        ax1.set_ylabel("Torque (Nm)")
        ax1.legend(loc='upper right', ncol=4, fontsize=8)
        ax1.grid(True, alpha=0.3)

        # 手臂（索引 13~22）
        ax2 = axes[1]
        for idx in range(13, 23):
            ax2.plot(time_arr, torque_arr[:, idx], label=joint_names[idx], linewidth=0.8)
        ax2.set_ylabel("Torque (Nm)")
        ax2.legend(loc='upper right', ncol=5, fontsize=8)
        ax2.grid(True, alpha=0.3)

        # 头部（索引 23,24）
        ax3 = axes[2]
        for idx in range(23, 25):
            ax3.plot(time_arr, torque_arr[:, idx], label=joint_names[idx], linewidth=0.8)
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Torque (Nm)")
        ax3.legend(loc='upper right', fontsize=8)
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot', type=str, default="casbot")
    parser.add_argument('--record_video', action='store_true')
    args = parser.parse_args()

    checkpoint = "/home/casbot/Desktop/robot_lab/logs/rsl_rl/casbot_02_adaptivemimic_flat/2026-04-10_13-42-26/model_best_reward.pt"
    motion_file = "/home/casbot/Desktop/hjq/motion/retarget_motion/lafan_casbot/fallAndGetUp2_subject2_clip3.npz"

    env = HumanoidEnv(
        policy_path=checkpoint,
        motion_path=motion_file,
        robot_type=args.robot,
        device="cpu",
        record_video=args.record_video
    )
    env.run()