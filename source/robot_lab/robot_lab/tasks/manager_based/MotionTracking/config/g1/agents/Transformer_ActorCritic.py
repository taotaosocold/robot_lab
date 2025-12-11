import torch
import torch.nn as nn
from tensordict import TensorDict
from rsl_rl.modules.actor_critic import ActorCritic
from torch.distributions import Normal

def get_proprio_dim(obs: TensorDict, obs_groups: dict[str, list[str]]) -> int:
    proprio_dim = 0
    for key in obs_groups["policy"]:
        if "proprio" in key:
            proprio_dim += obs[key].shape[-1]
    return proprio_dim

def get_motion_future_dim(obs: TensorDict, obs_groups: dict[str, list[str]]) -> int:
    motion_future_dim = 0
    for key in obs_groups["policy"]:
        if "future_motion" in key:
            motion_future_dim += obs[key].shape[-1]
    return motion_future_dim

def get_future_steps(obs: TensorDict, obs_groups: dict[str, list[str]]) -> int:
    for key in obs_groups["policy"]:
        if "future_motion" in key:
            return obs[key].shape[1]
    raise ValueError("No future_motion key found in obs_groups['policy']")

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
        self.proprio_dim = get_proprio_dim(obs, obs_groups)
        self.motion_future_dim = get_motion_future_dim(obs, obs_groups)
        self.future_steps = get_future_steps(obs, obs_groups)
        self.d_model = d_model

        self.motion_proj = nn.Linear(self.motion_future_dim, d_model)
        self.proprio_proj = nn.Linear(self.proprio_dim, d_model)

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
            nn.Linear(d_model + self.proprio_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 1),
        )

        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))

        self.action_mean = None
        self.action_std = None
        self.entropy = None

    def _forward_transformer(self, observations: TensorDict):
        future_motion = observations["future_motion"] # (B, future_steps, future_dim)
        proprio = observations["proprio"]             # (B, proprio_dim)
        B = future_motion.shape[0]

        # Encoder Logic
        future_emb = self.motion_proj(future_motion) + self.future_pos_emb
        memory = self.encoder(future_emb) # (B, future_steps, d_model)

        # Decoder Logic
        proprio_emb = self.proprio_proj(proprio).unsqueeze(1) # (B, 1, d_model)
        query = self.query_embed.expand(B, 1, -1)             # (B, 1, d_model)
        tgt = query + proprio_emb
        
        decoder_out = self.decoder(tgt=tgt, memory=memory)    # (B, 1, d_model)
        latent = decoder_out.squeeze(1)                       # (B, d_model)
        
        return latent, memory, proprio

    def act(self, observations: TensorDict, **kwargs):
        latent, _, _ = self._forward_transformer(observations)
        self.action_mean = self.actor_head(latent)
        self.action_std = self.std.expand_as(self.action_mean)

        dist = Normal(self.action_mean, self.action_std)
        self.entropy = dist.entropy().sum(dim=-1)
        return dist.sample()

    def get_value(self, observations: TensorDict) -> torch.Tensor:
        future_motion = observations["future_motion"]
        proprio = observations["proprio"]

        future_emb = self.motion_proj(future_motion) + self.future_pos_emb
        memory = self.encoder(future_emb)                          # (B, 32, D)
        future_summary = memory.mean(dim=1)                         # (B, D)  ← 全局池化

        x = torch.cat([future_summary, proprio], dim=-1)           # (B, D + p_dim)
        return self.critic_head(x)

    def get_actions_log_prob(self, actions):
        dist = Normal(self.action_mean, self.action_std)
        return dist.log_prob(actions).sum(dim=-1)

    def evaluate(self, observations: TensorDict, **kwargs) -> torch.Tensor:
        latent, memory, proprio = self._forward_transformer(observations)

        future_summary = memory.mean(dim=1) # (B, d_model)
        
        x = torch.cat([future_summary, proprio], dim=-1)
        return self.critic_head(x)

    def act_inference(self, observations: TensorDict):
        with torch.no_grad():
            latent, _, _ = self._forward_transformer(observations)
            return self.actor_head(latent)
    
    def reset(self, env_ids=None):
        pass

    def update_normalization(self, obs: TensorDict) -> None:
        pass

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict)
        return True