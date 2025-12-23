# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from .Transformer_ActorCritic import TransformerEncoderDecoderActorCritic

@configclass
class TransformerEncoderDecoderActorCriticCfg:
    class_name: str = "__import__('robot_lab.tasks.manager_based.MotionTracking.config.g1.agents.Transformer_ActorCritic', fromlist=['TransformerEncoderDecoderActorCritic']).TransformerEncoderDecoderActorCritic"
    d_model: int = 512
    nhead: int = 4
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    dim_feedforward: int = 1024
    dropout: float = 0.0    
    init_noise_std: float = 0.2

@configclass
class TransformerEncoderActorCriticCfg:
    class_name: str = "__import__('robot_lab.tasks.manager_based.MotionTracking.config.g1.agents.Transformer_ActorCritic', fromlist=['TransformerEncoderActorCritic']).TransformerEncoderActorCritic"
    d_model: int = 128
    nhead: int = 2
    num_encoder_layers: int = 4
    dim_feedforward: int = 256
    dropout: float = 0.0
    init_noise_std: float = 0.2

@configclass
class TransformerEncoderMLPActorCriticCfg:
    class_name: str = "__import__('robot_lab.tasks.manager_based.MotionTracking.config.g1.agents.Transformer_ActorCritic', fromlist=['TransformerEncoderMLPActorCritic']).TransformerEncoderMLPActorCritic"
    d_model: int = 128
    nhead: int = 1
    num_encoder_layers: int = 2
    dim_feedforward: int = 128
    mlp_hidden_dims: list[int] = [512, 256, 128]
    activation: str = "elu"
    dropout: float = 0.0
    init_noise_std: float = 0.2

@configclass
class UnitreeG1MotionTrackingFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 100
    experiment_name = "unitree_g1_MotionTracking_flat"
    obs_groups = {
        "policy": ["proprio_history", "future_motion"],
        "critic": ["proprio_history", "future_motion"],
    }
    policy = TransformerEncoderActorCriticCfg()
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=32,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    # eval
    load_run = "2025-12-22_17-55-45"
    load_checkpoint = "model_2300.pt"
