# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""OnPolicyRunner extension that additionally saves the best-reward checkpoint."""

import os
import statistics

import torch
from rsl_rl.runners import OnPolicyRunner


class OnPolicyRunner(OnPolicyRunner):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._best_mean_reward = float("-inf")
        self._best_mean_reward_iter = 0
        self._best_mean_ep_len = float("-inf")
        self._best_mean_ep_len_iter = 0

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:  # noqa: F811
        import time

        from rsl_rl.utils import check_nan

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()

        if self.is_distributed:
            self.alg.broadcast_parameters()

        self.logger.init_logging_writer()

        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations

        for it in range(start_it, total_it):
            start = time.time()

            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    if self.cfg.get("check_for_nan", True):
                        check_nan(obs, rewards, dones)
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.cfg["algorithm"]["rnd_cfg"] else None
                    self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)

                stop = time.time()
                collect_time = stop - start
                start = stop
                self.alg.compute_returns(obs)

            loss_dict = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.get_policy().output_std,
                rnd_weight=self.alg.rnd.weight if self.cfg["algorithm"]["rnd_cfg"] else None,
            )

            if self.logger.writer is not None:
                # periodic checkpoint
                if it % self.cfg["save_interval"] == 0:
                    self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))

                # best-reward checkpoint
                if len(self.logger.rewbuffer) > 0:
                    mean_reward = statistics.mean(self.logger.rewbuffer)
                    if mean_reward > self._best_mean_reward:
                        self._best_mean_reward = mean_reward
                        self._best_mean_reward_iter = it
                        self.save(os.path.join(self.logger.log_dir, "model_best_reward.pt"))

                # best-episode-length checkpoint
                if len(self.logger.lenbuffer) > 0:
                    mean_ep_len = statistics.mean(self.logger.lenbuffer)
                    if mean_ep_len > self._best_mean_ep_len:
                        self._best_mean_ep_len = mean_ep_len
                        self._best_mean_ep_len_iter = it
                        self.save(os.path.join(self.logger.log_dir, "model_best_ep_len.pt"))

                print(f"{'Best mean reward:'} {self._best_mean_reward:.2f}  (iter {self._best_mean_reward_iter})")
                print(f"{'Best mean episode len:'} {self._best_mean_ep_len:.2f}  (iter {self._best_mean_ep_len_iter})")

        if self.logger.writer is not None:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))
            self.logger.stop_logging_writer()
