import torch
import torch.nn as nn
from tensordict import TensorDict
from rsl_rl.modules.actor_critic import ActorCritic

# No History
class TransformerEncoderDecoderActorCritic(ActorCritic):
    def __init__(
        self,
        num_actions: int,
        proprio_dim: int,
        motion_future_dim: int,
        future_steps: int,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        init_noise_std: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.future_steps = future_steps
        self.d_model = d_model

        self.motion_proj = nn.Linear(motion_future_dim, d_model)
        self.proprio_proj = nn.Linear(proprio_dim, d_model)

        self.future_pos_emb = nn.Parameter(torch.randn(1, self.future_steps, d_model) * 0.02)

        self.query_embed = nn.Parameter(torch.randn(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)


        self.actor_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, num_actions),
        )

        self.critic_head = nn.Sequential(
            nn.Linear(d_model + proprio_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 1),
        )

        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))

    def act(self, observations: TensorDict):
        future_motion = observations["future_motion"]      # (B, future_steps, future_dim)
        proprio   = observations["proprio"]      # (B, proprio_dim)

        B = future_motion.shape[0]
        memory = self.forward_encoder(future, hist, curr)     # (B, 71, D)

        future_emb = self.motion_proj(future_motion) + self.future_pos_emb
        memory = self.encoder(future_emb)

        proprio_emb = self.proprio_proj(proprio).unsqueeze(1)   #(B, 1, D)
        query = self.query_embed.expand(B, 1, -1)            # (B, 1, D)
        tgt = query + curr_emb

        decoder_out = self.decoder(tgt=tgt, memory=memory)
        latent = decoder_out.squeeze(1)                       # (B, D)
        action_mean = self.actor_head(latent)
        return action_mean

    def get_value(self, observations: TensorDict) -> torch.Tensor:
        future_motion = observations["future_motion"]
        proprio = observations["proprio"]

        future_emb = self.motion_proj(future_motion) + self.future_pos_emb
        memory = self.encoder(future_emb)                          # (B, 32, D)
        future_summary = memory.mean(dim=1)                         # (B, D)  ← 全局池化

        x = torch.cat([future_summary, proprio], dim=-1)           # (B, D + p_dim)
        return self.critic_head(x)

    def act_inference(self, observations: TensorDict):
        with torch.no_grad():
            return self.act(observations)

    def reset(self, env_ids=None):
        pass