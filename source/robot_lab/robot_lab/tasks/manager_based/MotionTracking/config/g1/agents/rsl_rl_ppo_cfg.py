# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from .Transformer_ActorCritic import TransformerEncoderDecoderActorCritic

@configclass
class TransformerActorCriticCfg:
    class_name = TransformerEncoderDecoderActorCritic
    num_actions: int = 23
    proprio_dim: int = 3+3+23+23+23          # 78
    motion_future_dim: int = 14*3+14*3+14*3+14*3+23+23   # 214    
    future_steps: int = 32
    d_model: int = 512
    nhead: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.1    
    init_noise_std: float = 1.0



@configclass
class UnitreeG1MotionTrackingFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 500
    experiment_name = "unitree_g1_MotionTracking_flat"
    policy = TransformerActorCriticCfg()
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
