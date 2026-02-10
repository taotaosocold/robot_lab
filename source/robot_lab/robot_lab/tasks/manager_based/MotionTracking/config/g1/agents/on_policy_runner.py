import os
import statistics
import torch
from rsl_rl.runners import OnPolicyRunner

class OnPolicyRunner(OnPolicyRunner):
    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        super().__init__(env, train_cfg, log_dir, device)
        self.best_mean_episode_length = -1.0
        self.best_iter = 0

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        self._prepare_logging_writer()
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations().to(self.device)
        self.train_mode()

        from collections import deque
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100) # 我们需要监视这个 buffer
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        if self.alg.rnd:
            erewbuffer = deque(maxlen=100)
            irewbuffer = deque(maxlen=100)
            cur_ereward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            cur_ireward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations

        for it in range(start_iter, tot_iter):
            import time
            start = time.time()
            # Rollout 阶段
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    
                    if self.log_dir is not None:
                        if "episode" in extras:
                            ep_infos.append(extras["episode"])
                        elif "log" in extras:
                            ep_infos.append(extras["log"])
                        
                        if self.alg.rnd:
                            cur_ereward_sum += rewards
                            cur_ireward_sum += self.alg.intrinsic_rewards
                            cur_reward_sum += rewards + self.alg.intrinsic_rewards
                        else:
                            cur_reward_sum += rewards
                        
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                        if self.alg.rnd:
                            erewbuffer.extend(cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            irewbuffer.extend(cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            cur_ereward_sum[new_ids] = 0
                            cur_ireward_sum[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                self.alg.compute_returns(obs)

            # 更新策略
            loss_dict = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            # --- 关键修改：判断并保存最佳模型 ---
            if len(lenbuffer) > 0:
                current_mean_len = statistics.mean(lenbuffer)
                if current_mean_len > self.best_mean_episode_length:
                    self.best_mean_episode_length = current_mean_len
                    self.best_iter = it
                    if self.log_dir is not None and not self.disable_logs:
                        best_path = os.path.join(self.log_dir, "best_length.pt")
                        self.save(best_path)
                        print(f"\n[Best Model] 找到更好的模型！已保存至 {best_path}")
                        print(f"[Best Model] 当前最高平均步数: {self.best_mean_episode_length:.2f} (Iteration: {it})\n")

            # 原有的保存和日志逻辑
            if self.log_dir is not None and not self.disable_logs:
                self.log(locals()) # 这里的 locals 会把 best_mean_episode_length 传进去
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            ep_infos.clear()
            # 存储代码状态等（略）...

    def log(self, locs: dict, width: int = 80, pad: int = 35) -> None:
        super().log(locs, width, pad)
        
        if not self.disable_logs:
            best_info = (
                f"""{"-" * width}\n"""
                f"""{f"Best Mean Episode Length:":>{pad}} {self.best_mean_episode_length:.2f}\n"""
                f"""{f"Best Model Iteration:":>{pad}} {self.best_iter}\n"""
                f"""{"-" * width}\n"""
            )
            print(best_info)
            if self.writer is not None:
                self.writer.add_scalar("Train/best_episode_length", self.best_mean_episode_length, locs["it"])