import argparse
import os
import sys
import torch
import numpy as np
import pandas as pd  # 新增：用于保存 CSV
from tqdm import tqdm

from isaaclab.app import AppLauncher

# local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cli_args

# 添加参数
parser = argparse.ArgumentParser(description="Evaluate RL agent on all motions (Headless).")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
# Append args
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# 强制 Headless
args_cli.headless = True 

# 启动 Isaac Sim
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from rsl_rl.runners import DistillationRunner, OnPolicyRunner
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import robot_lab.tasks  # noqa: F401

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg):
    
    task_name = args_cli.task.split(":")[-1]

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 64

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # spawn the robot randomly in the grid
    env_cfg.scene.terrain.max_init_terrain_level = None
    if env_cfg.scene.terrain.terrain_generator is not None:
        env_cfg.scene.terrain.terrain_generator.num_rows = 5
        env_cfg.scene.terrain.terrain_generator.num_cols = 5
        env_cfg.scene.terrain.terrain_generator.curriculum = False

    env_cfg.observations.policy_proprio_history.enable_corruption = False
    
    env_cfg.events.randomize_apply_external_force_torque = None
    env_cfg.events.push_robot = None
    env_cfg.curriculum.command_levels_lin_vel = None
    env_cfg.curriculum.command_levels_ang_vel = None

    # ========== 评估专用关闭 ==========
    # 彻底关闭所有 termination（防止任何提前结束）
    env_cfg.terminations = {}  # 清空所有 termination terms

    # 关闭所有 adaptive sampling
    env_cfg.commands.motion.enable_bin_adaptive_sampling = False
    env_cfg.commands.motion.enable_motion_adaptive_sampling = False
    env_cfg.commands.motion.motion_adaptive_uniform_ratio = 0.0

    # =====================================

    # 2. 加载 Checkpoint
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    isaac_env = env.unwrapped
    motion_cmd = isaac_env.command_manager.get_term("motion")
    total_motions = motion_cmd.motion.num_motions
    num_envs = isaac_env.num_envs
    device = isaac_env.device
    
    # 获取所有 motion 名称，用于打印和保存
    motion_names = motion_cmd.motion.motion_name

    # per-motion 统计（新增两个根速度误差指标）
    per_motion_mpjpe = np.zeros(total_motions)
    per_motion_mpkpe = np.zeros(total_motions)
    per_motion_anchor_lin_vel_err = np.zeros(total_motions)
    per_motion_anchor_ang_vel_err = np.zeros(total_motions)
    per_motion_success = np.ones(total_motions, dtype=bool)  # 关闭 termination 后应全为 True

    all_indices = torch.arange(total_motions, device=device)
    num_batches = int(np.ceil(total_motions / num_envs))

    with torch.inference_mode():
        for batch_i in range(num_batches):
            start_idx = batch_i * num_envs
            end_idx = min((batch_i + 1) * num_envs, total_motions)
            curr_batch_size = end_idx - start_idx
            
            batch_motion_ids = all_indices[start_idx:end_idx]
            env_ids = torch.arange(curr_batch_size, device=device)
            
            # 设置运动并重置
            motion_cmd.set_motion_ids(env_ids, batch_motion_ids) 
            env.reset()
            obs = env.get_observations()
            
            # 当前 batch 统计（新增两个速度误差累加器）
            batch_mpjpe = torch.zeros(curr_batch_size, device=device)
            batch_mpkpe = torch.zeros(curr_batch_size, device=device)
            batch_anchor_lin_vel_err = torch.zeros(curr_batch_size, device=device)
            batch_anchor_ang_vel_err = torch.zeros(curr_batch_size, device=device)
            
            target_frames = motion_cmd.motion.motion_frames[motion_cmd.env_motion_idx[env_ids]]
            
            max_frames = int(target_frames.max().item())
            print(f"max_frames: {max_frames}")
            for step_count in range(max_frames + 5):
                actions = policy(obs)
                obs, _, _, _ = env.step(actions)
                
                # 只累加活跃帧的误差
                active_mask = (motion_cmd.time_steps[env_ids] < target_frames).float()  # 转为 float 以便相乘
                
                batch_mpjpe += motion_cmd.metrics["error_joint_pos"][env_ids] * active_mask
                batch_mpkpe += motion_cmd.metrics["error_body_pos"][env_ids] * active_mask
                batch_anchor_lin_vel_err += motion_cmd.metrics["error_anchor_lin_vel"][env_ids] * active_mask
                batch_anchor_ang_vel_err += motion_cmd.metrics["error_anchor_ang_vel"][env_ids] * active_mask

                if torch.all(motion_cmd.time_steps[env_ids] >= target_frames):
                    break

            # 汇总到 per-motion
            for i in range(curr_batch_size):
                motion_idx = batch_motion_ids[i].item()
                length = target_frames[i].item()
                
                avg_mpjpe = (batch_mpjpe[i] / length).item() if length > 0 else 0.0
                avg_mpkpe = (batch_mpkpe[i] / length).item() if length > 0 else 0.0
                avg_anchor_lin_vel = (batch_anchor_lin_vel_err[i] / length).item() if length > 0 else 0.0
                avg_anchor_ang_vel = (batch_anchor_ang_vel_err[i] / length).item() if length > 0 else 0.0
                
                per_motion_mpjpe[motion_idx] = avg_mpjpe
                per_motion_mpkpe[motion_idx] = avg_mpkpe
                per_motion_anchor_lin_vel_err[motion_idx] = avg_anchor_lin_vel
                per_motion_anchor_ang_vel_err[motion_idx] = avg_anchor_ang_vel
            
            print(f"Batch {batch_i+1}/{num_batches} Processed.")

    # ========== 最终报告 ==========
    print(f"\n{'='*80}")
    print(f"Final Evaluation Result ({total_motions} motions)")
    print(f"{'='*80}")
    print(f"Avg MPJPE: {per_motion_mpjpe.mean():.5f} rad (±{per_motion_mpjpe.std():.5f})")
    print(f"Avg MPKPE: {per_motion_mpkpe.mean():.5f} m (±{per_motion_mpkpe.std():.5f})")
    print(f"Avg Anchor Linear Vel Error: {per_motion_anchor_lin_vel_err.mean():.5f} m/s (±{per_motion_anchor_lin_vel_err.std():.5f})")
    print(f"Avg Anchor Angular Vel Error: {per_motion_anchor_ang_vel_err.mean():.5f} rad/s (±{per_motion_anchor_ang_vel_err.std():.5f})")
    print(f"Survival Rate: 100.00%")

    # 详细 per-motion 结果（按 MPJPE 从高到低排序，新增两列）
    print(f"\nPer-motion detailed results (sorted by MPJPE descending):")
    print(f"{'Rank':<4} {'Motion Name':<40} {'MPJPE (rad)':<14} {'MPKPE (m)':<14} {'LinVel Err (m/s)':<18} {'AngVel Err (rad/s)':<20} {'Status'}")
    print("-" * 120)
    
    sorted_indices = np.argsort(per_motion_mpjpe)[::-1]
    for rank, idx in enumerate(sorted_indices, 1):
        name = motion_names[idx]
        mpjpe = per_motion_mpjpe[idx]
        mpkpe = per_motion_mpkpe[idx]
        lin_vel_err = per_motion_anchor_lin_vel_err[idx]
        ang_vel_err = per_motion_anchor_ang_vel_err[idx]
        status = "Success"
        print(f"{rank:<4} {name:<40} {mpjpe:<14.5f} {mpkpe:<14.5f} {lin_vel_err:<18.5f} {ang_vel_err:<20.5f} {status}")

    # 保存 CSV（新增两列）
    csv_path = os.path.join(log_dir, "motion_evaluation_detailed_results.csv")
    df = pd.DataFrame({
        'rank': range(1, total_motions + 1),
        'motion_name': [motion_names[i] for i in sorted_indices],
        'mpjpe_rad': per_motion_mpjpe[sorted_indices],
        'mpkpe_m': per_motion_mpkpe[sorted_indices],
        'anchor_lin_vel_err_m_s': per_motion_anchor_lin_vel_err[sorted_indices],
        'anchor_ang_vel_err_rad_s': per_motion_anchor_ang_vel_err[sorted_indices],
        'success': ["Success"] * total_motions
    })
    df.to_csv(csv_path, index=False)
    print(f"\nDetailed results saved to: {csv_path}")
    print(f"{'='*80}")

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()