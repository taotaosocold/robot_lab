import torch
import torch.nn as nn
from tensordict import TensorDict
from rsl_rl.modules.actor_critic import ActorCritic
from torch.distributions import Normal
from rsl_rl.networks import EmpiricalNormalization
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: [Batch, Seq_Len, Dim]
        return x + self.pe[:, :x.size(1), :]

# No History
class TransformerEncoderDecoderActorCritic(nn.Module):
    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        init_noise_std: float,
        **kwargs,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.proprio_dim = self._get_dim(obs, obs_groups, "proprio")
        self.motion_future_dim = self._get_dim(obs, obs_groups, "future_motion")
        self.future_steps = self._get_steps(obs, obs_groups)
        self.d_model = d_model

        self.motion_proj = nn.Linear(self.motion_future_dim, d_model)
        self.proprio_proj = nn.Linear(self.proprio_dim, d_model)

        self.pos_encoder = PositionalEncoding(d_model, max_len=self.future_steps)

        self.proprio_norm = EmpiricalNormalization((self.proprio_dim,))
        self.motion_future_norm = EmpiricalNormalization((self.motion_future_dim,))

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
            nn.Linear(d_model, num_actions),
        )

        self.critic_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Linear(256, 1),
        )

        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self._init_weights()
        self.action_mean = None
        self.action_std = None

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # Actor 最后一层初始化为接近 0
        nn.init.constant_(self.actor_head[-1].weight, 0.01)
        nn.init.constant_(self.actor_head[-1].bias, 0.0)
    def _forward_transformer(self, observations: TensorDict):
        future_motion = observations["future_motion"] # (B, future_steps, future_dim)
        proprio = observations["proprio"]             # (B, proprio_dim)
        B = future_motion.shape[0]

        future_motion = self.motion_future_norm(future_motion) # Normalize future motion
        proprio = self.proprio_norm(proprio)                   # Normalize proprioception

        # Encoder Logic
        future_motion_emb = self.motion_proj(future_motion)
        future_motion_emb = self.pos_encoder(future_motion_emb)
        memory = self.encoder(future_motion_emb) # (B, future_steps, d_model)

        # Decoder Logic
        proprio_emb = self.proprio_proj(proprio).unsqueeze(1) # (B, 1, d_model)
        query = self.query_embed.expand(B, 1, -1) + proprio_emb # (B, 1, d_model)
        
        decoder_out = self.decoder(tgt=query, memory=memory)    # (B, 1, d_model)
        latent = decoder_out.squeeze(1)                       # (B, d_model)
        
        return latent

    def act(self, observations: TensorDict, **kwargs):
        latent = self._forward_transformer(observations)
        self.action_mean = self.actor_head(latent)
        self.action_std = self.std.expand_as(self.action_mean)

        dist = Normal(self.action_mean, self.action_std)
        if self.training:
            self.entropy = dist.entropy().sum(dim=-1)
        return dist.sample()

    def get_value(self, observations: TensorDict) -> torch.Tensor:
        latent = self._forward_transformer(observations)
        return self.critic_head(latent)


    def get_actions_log_prob(self, actions):
        dist = Normal(self.action_mean, self.action_std)
        return dist.log_prob(actions).sum(dim=-1)

    def evaluate(self, observations: TensorDict, **kwargs) -> torch.Tensor:
        latent = self._forward_transformer(observations)
        return self.critic_head(latent)

    def act_inference(self, observations: TensorDict):
        with torch.no_grad():
            latent = self._forward_transformer(observations)
            return self.actor_head(latent)
    
    def reset(self, env_ids=None):
        pass

    def update_normalization(self, obs: TensorDict) -> None:
        if "future_motion" in obs:
            future_motion = obs["future_motion"].view(-1, fm.shape[-1])
            self.motion_future_norm.update(future_motion)
        if "proprio" in obs:
            self.proprio_norm.update(obs["proprio"])

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict)
        return True

    def _get_dim(self, obs, groups, keyword):
        dim = 0
        for key in groups["policy"]:
            if keyword in key:
                dim += obs[key].shape[-1]
        return dim

    def _get_steps(self, obs, groups):
        for key in groups["policy"]:
            if "future_motion" in key:
                return obs[key].shape[1]
        return 0