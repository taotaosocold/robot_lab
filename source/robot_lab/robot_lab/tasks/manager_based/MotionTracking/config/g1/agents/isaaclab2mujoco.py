import argparse, os, time
import numpy as np
import mujoco, mujoco_viewer
from tqdm import tqdm
from collections import deque
import torch
from Transformer_ActorCritic import TransformerEncoderDecoderActorCritic, TransformerEncoderDecoderActorCritic
from tensordict import TensorDict
import isaaclab.utils.math as math_utils

class HumanoidEnv:
    def __init__(self, policy_path, motion_path, robot_type="g1", device="cuda", record_video=False):
        self.robot_type = robot_type
        self.device = device
        self.record_video = record_video
        self.motion_path = motion_path
        self.policy_path = policy_path

        self.load_motion()
        self.load_model()
        self.dof_index = [0, 6, 12, 1, 7, 13, 18, 2, 8, 14, 19, 3, 9, 15, 20, 4, 10, 16, 21, 5, 11, 17, 22]
        self.body_index = [0, 5, 15, 23, 6, 16, 24, 4, 13, 21, 25, 14, 22, 26]
        if robot_type == "g1":
            model_path = "/home/ubuntu/Desktop/hjq/assets/g1_23dof.xml"
            self.stiffness = np.array([
                100, 100, 100, 200, 20, 20,
                100, 100, 100, 200, 20, 20,
                400,
                90, 60, 20, 60, 60,
                90, 60, 20, 60, 60
            ])
            self.damping = np.array([
                2.5, 2.5, 2.5, 5.0, 0.2, 0.1,
                2.5, 2.5, 2.5, 5.0, 0.2, 0.1,
                5.0,
                2.0, 1.0, 0.4, 1.0, 1.0,
                2.0, 1.0, 0.4, 1.0, 1.0
            ])
            self.num_actions = 23
            self.num_dofs = 23
            self.default_dof_pos = np.array([
                -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,  # left leg (6)
                -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,  # right leg (6)
                0.0,
                0.2, 0.2, 0.0, 0.6, 0.0,
                0.2, -0.2, 0.0, 0.6, 0.0,
            ])
            self.torque_limits = np.array([
                88, 88, 88, 139, 50, 50,
                88, 88, 88, 139, 50, 50,
                88,
                25, 25, 25, 25,
                25, 25, 25, 25,
            ])
        else:
            raise ValueError(f"Robot type {robot_type} not supported!")
        
        self.obs_indices = np.arange(self.num_dofs)
        
        self.sim_duration = 60.0
        self.sim_dt = 0.001
        self.sim_decimation = 20
        self.control_dt = self.sim_dt * self.sim_decimation
        
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
        
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)

        self.tar_obs_steps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                         11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
        
        self._init_motion_buffers()
        
        print("Loading jit for policy: ", policy_path)
        self.policy_path = policy_path

        
        self.last_time = time.time()

    def load_motion(self):
        data = np.load(self.motion_path, allow_pickle=True)
        self.joint_pos = torch.tensor(data['joint_pos'], device=self.device, dtype=torch.float32)
        self.joint_vel = torch.tensor(data['joint_vel'], device=self.device, dtype=torch.float32)
        self.body_pos_w = torch.tensor(data['body_pos_w'], device=self.device, dtype=torch.float32)
        self.body_quat_w = torch.tensor(data['body_quat_w'], device=self.device, dtype=torch.float32)
        self.body_lin_vel_w = torch.tensor(data['body_lin_vel_w'], device=self.device, dtype=torch.float32)
        self.body_ang_vel_w = torch.tensor(data['body_ang_vel_w'], device=self.device, dtype=torch.float32)
        self.body_pos_r = torch.tensor(data['body_pos_r'], device=self.device, dtype=torch.float32)
        self.body_quat_r = torch.tensor(data['body_quat_r'], device=self.device, dtype=torch.float32)
        self.body_lin_vel_r = torch.tensor(data['body_lin_vel_r'], device=self.device, dtype=torch.float32)
        self.body_ang_vel_r = torch.tensor(data['body_ang_vel_r'], device=self.device, dtype=torch.float32)
        self.motion_len = self.joint_pos.shape[0]

    def load_model(self):
        print(f"Loading policy from {self.policy_path}")
        self.policy_cfg = {
            "num_actions": self.num_actions,
            "d_model": 256,
            "nhead": 4,
            "num_encoder_layers": 4,
            "dim_feedforward": 1024,
            "dropout": 0.0,
            "init_noise_std": 1.0
        }
        num_bodies = 14 
        self.proprio_dim = 3 + 3 + 3 + 23 + 23 + 23
        self.future_motion_dim = (num_bodies * 3) + (num_bodies * 6) + (num_bodies * 3) + (num_bodies * 3) + 23 + 23
        dummy_obs = TensorDict({
            "proprio": torch.zeros(1, self.proprio_dim),
            "future_motion": torch.zeros(1, self.future_steps, self.future_motion_dim)
        }, batch_size=[1])
        
        obs_groups = {"policy": ["proprio", "future_motion"]}
        self.policy_net = TransformerEncoderActorCritic(
            obs=dummy_obs,
            obs_groups=obs_groups,
            **self.policy_cfg
        )
        self.policy_net.load_state_dict(state_dict)
        self.policy_net.to(self.device)
        self.policy_net.eval()

    def _init_motion_buffers(self):
        self.tar_obs_steps = torch.tensor(self.tar_obs_steps, device=self.device, dtype=torch.int)

    def get_future_obs(self, curr_time_step):
        target_indices = self.tar_obs_steps + curr_time_step
        target_indices = torch.clamp(target_indices, max=self.motion_len - 1)

        body_pos_r = self.body_pos_r[:, self.body_index][target_indices]
        body_quat_r = self.body_quat_r[:, self.body_index][target_indices]
        body_lin_vel_r = self.body_lin_vel_r[:, self.body_index][target_indices]
        body_ang_vel_r = self.body_ang_vel_r[:, self.body_index][target_indices]
        joint_pos = self.joint_pos[:, self.body_index][target_indices]
        joint_vel = self.joint_vel[:, self.body_index][target_indices]

        body_ori_r = math_utils.matrix_from_quat(body_quat_r.view(-1, 4))[..., :2]
        
        future_obs = []
        future_obs.append(body_pos_r)
        future_obs.append(body_ori_r)
        future_obs.append(body_lin_vel_r)
        future_obs.append(body_ang_vel_r)
        future_obs.append(joint_pos)
        future_obs.append(joint_vel)
        future_obs = torch.cat(future_obs, dim=-1)
        
        return future_obs.flatten()
        
    def get_proprio_obs(self):
        dof_pos = self.data.qpos.astype(np.float32)[-self.num_dofs:]
        dof_vel = self.data.qvel.astype(np.float32)[-self.num_dofs:]
        anchor_quat_w = self.data.sensor('imu-torso-quat').data.astype(np.float32)
        base_lin_vel_r = self.data.sensor('imu-torso-lin-vel-r').data.astype(np.float32)
        base_ang_vel_r = self.data.sensor('imu-torso-angular-velocity').data.astype(np.float32)
        gravity_vec_w = torch.tensor([0.0, 0.0, -1.0], device=device).repeat(batch_size, 1)
        projected_gravity_b = math_utils.quat_apply_inverse(anchor_quat_w, gravity_vec_w)
        dof_pos = dof_pos - self.default_dof_pos
        dof_vel = dof_vel - 0
        dof_pos = dof_pos[self.dof_index]
        dof_vel = dof_vel[self.dof_index]
        return (base_lin_vel_r, base_ang_vel_r, projected_gravity_b, dof_pos, dof_vel)
        
    def run(self):
        motion_name = os.path.basename(self.motion_path).split('.')[0]
        if self.record_video:
            import imageio
            video_name = f"{self.robot_type}_{''.join(os.path.basename(self.policy_path).split('.')[:-1])}_{motion_name}.mp4"
            path = "mujoco_videos/"
            if not os.path.exists(path):
                os.makedirs(path)
            video_name = os.path.join(path, video_name)
            mp4_writer = imageio.get_writer(video_name, fps=50)
        
        for i in tqdm(range(int(self.sim_duration / self.sim_dt)), desc="Running simulation..."):
            # 从mujoco中获得机器人本体数据
            base_lin_vel_r, base_ang_vel_r, projected_gravity_b, dof_pos, dof_vel = self.get_proprio_obs()

            if i % self.sim_decimation == 0:
                curr_timestep = i // self.sim_decimation
                future_obs = self.get_future_obs(curr_timestep).unsqueeze(0)
                
                prop_obs = np.concatenate([
                    base_lin_vel_r,
                    base_ang_vel_r,
                    projected_gravity_b,
                    dof_pos,
                    dof_vel
                    self.last_action,
                ])

                prop_obs = torch.from_numpy(prop_obs).float().to(self.device).unsqueeze(0)
                
                obs_dict = TensorDict({
                    "proprio": prop_obs,
                    "future_motion": future_obs
                }, batch_size=[1])
                
                raw_action = self.policy_net.act_inference(obs_dict).cpu().numpy().squeeze()
                
                self.last_action = raw_action.copy()
                raw_action = np.clip(raw_action, -10., 10.)
                
                step_actions = np.zeros(self.num_dofs)
                step_actions = raw_action
                
                pd_target = step_actions
                
                self.viewer.cam.lookat = self.data.qpos.astype(np.float32)[:3]
                if self.record_video:
                    img = self.viewer.read_pixels()
                    mp4_writer.append_data(img)
                else:
                    self.viewer.render()
                
        
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
    checkpoint = "/home/ubuntu/Desktop/robot_lab/logs/rsl_rl/unitree_g1_MotionTracking_flat/2025-12-18_14-07-47/model_200.pt"
    motion_file = "/home/ubuntu/Desktop/humanoid-general-motion-tracking/assets/motion/B3 - walk1_poses.npz"
    assert os.path.exists(checkpoint), f"Policy path {checkpoint} does not exist!"
    print(f"Loading model from: {checkpoint}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    env = HumanoidEnv(policy_path=checkpoint, motion_path=motion_file, robot_type=args.robot, device=device, record_video=args.record_video)
    
    env.run()
        
        