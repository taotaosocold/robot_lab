import torch
import torch.nn as nn
from tensordict import TensorDict
from rsl_rl.modules.actor_critic import ActorCritic
from torch.distributions import Normal
from rsl_rl.networks import EmpiricalNormalization
import math
import numpy as np
import torch.nn.functional as F

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
        nn.init.constant_(self.actor_head[-1].weight, 1e-3)
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
            future_motion = obs["future_motion"]
            future_motion = future_motion.view(-1, future_motion.shape[-1])
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

    def _get_steps(self, obs, groups, keyword):
        for key in groups["policy"]:
            if keyword == "future_motion" in key:
                return obs[key].shape[1]
            if keyword == "proprio_history" in key:
                return obs[key].shape[1]
        return 0

# No History
class TransformerEncoderActorCritic(nn.Module):
    is_recurrent: bool = False

    def __init__(
        self, 
        obs: dict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        d_model: int, 
        nhead: int,
        num_encoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        init_noise_std: float,
        mlp_hidden_dims: list[int], 
        activation: str = "ELU",
        **kwargs,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.obs_groups = obs_groups
        self.d_model = d_model

        self.actor_proprio_dim = self._get_group_dim(obs, "policy", "proprio_history")
        self.actor_future_dim = self._get_group_dim(obs, "policy", "future_motion")
        self.critic_proprio_dim = self._get_group_dim(obs, "critic", "proprio_history")
        self.critic_future_dim = self._get_group_dim(obs, "critic", "future_motion")
        
        self.future_steps = self._get_group_steps(obs, "policy", "future_motion")
        self.history_steps = self._get_group_steps(obs, "policy", "proprio_history")

        self.actor_proprio_norm = EmpiricalNormalization((self.actor_proprio_dim,))
        self.critic_proprio_norm = EmpiricalNormalization((self.critic_proprio_dim,))
        self.motion_future_norm = EmpiricalNormalization((self.actor_future_dim,))

        self.pos_encoder = PositionalEncoding(d_model, max_len=self.history_steps + self.future_steps)
        self.register_buffer("type_0", torch.tensor(0, dtype=torch.long))
        self.register_buffer("type_1", torch.tensor(1, dtype=torch.long))
        self.modality_embed = nn.Embedding(2, d_model)

        self.actor_proprio_proj = nn.Linear(self.actor_proprio_dim, d_model)
        self.actor_future_proj = nn.Linear(self.actor_future_dim, d_model)
        actor_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.actor_encoder = nn.TransformerEncoder(actor_layer, num_layers=num_encoder_layers)
        self.actor_mlp = self._build_mlp(d_model, mlp_hidden_dims, num_actions, activation)

        self.critic_proprio_proj = nn.Linear(self.critic_proprio_dim, d_model)
        self.critic_future_proj = nn.Linear(self.critic_future_dim, d_model)
        critic_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.critic_encoder = nn.TransformerEncoder(critic_layer, num_layers=num_encoder_layers)
        self.critic_mlp = self._build_mlp(d_model, mlp_hidden_dims, 1, activation)
        
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.apply(self._init_weights)
        
        if isinstance(self.actor_mlp[-1], nn.Linear):
            nn.init.constant_(self.actor_mlp[-1].weight, 1e-3)
            nn.init.constant_(self.actor_mlp[-1].bias, 0.0)

    def _build_mlp(self, input_dim, hidden_dims, output_dim, activation_name):
        layers = []
        curr_dim = input_dim
        act_class = getattr(nn, activation_name.upper()) if isinstance(activation_name, str) else nn.ELU
        for h in hidden_dims:
            layers.append(nn.Linear(curr_dim, h))
            layers.append(act_class())
            curr_dim = h
        layers.append(nn.Linear(curr_dim, output_dim))
        return nn.Sequential(*layers)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def _forward_transformer(self, proprio, future, proprio_proj, future_proj, encoder):
        p_emb = proprio_proj(proprio) + self.modality_embed(self.type_0)
        f_emb = future_proj(future) + self.modality_embed(self.type_1)
        seq = torch.cat([p_emb, f_emb], dim=1)
        seq = self.pos_encoder(seq)
        out = encoder(seq)
        return out[:, self.history_steps - 1, :]

    def act(self, observations: dict, **kwargs):
        future = self.motion_future_norm(observations["future_motion"])
        proprio = self.actor_proprio_norm(observations["policy_proprio_history"])
        
        latent = self._forward_transformer(proprio, future, self.actor_proprio_proj, self.actor_future_proj, self.actor_encoder)
        self.action_mean = self.actor_mlp(latent)
        self.action_std = self.std.expand_as(self.action_mean)
        
        dist = Normal(self.action_mean, self.action_std)
        if self.training:
            self.entropy = dist.entropy().sum(dim=-1)
        return dist.sample()

    def get_value(self, observations: dict) -> torch.Tensor:
        future = self.motion_future_norm(observations["future_motion"])
        proprio = self.critic_proprio_norm(observations["critic_proprio_history"])
        
        latent = self._forward_transformer(proprio, future, self.critic_proprio_proj, self.critic_future_proj, self.critic_encoder)
        return self.critic_mlp(latent)

    def evaluate(self, observations: dict, **kwargs) -> torch.Tensor:
        return self.get_value(observations)

    def act_inference(self, observations: dict):
        with torch.no_grad():
            future = self.motion_future_norm(observations["future_motion"])
            proprio = self.actor_proprio_norm(observations["policy_proprio_history"])
            latent = self._forward_transformer(proprio, future, self.actor_proprio_proj, self.actor_future_proj, self.actor_encoder)
            return self.actor_mlp(latent)

    def update_normalization(self, obs: dict) -> None:
        if "future_motion" in obs:
            self.motion_future_norm.update(obs["future_motion"].view(-1, self.actor_future_dim))
        if "policy_proprio_history" in obs:
            self.actor_proprio_norm.update(obs["policy_proprio_history"].view(-1, self.actor_proprio_dim))
        if "critic_proprio_history" in obs:
            self.critic_proprio_norm.update(obs["critic_proprio_history"].view(-1, self.critic_proprio_dim))

    def _get_group_dim(self, obs, group_name, keyword):
        for key in self.obs_groups.get(group_name, []):
            if keyword in key:
                return obs[key].shape[-1]
        raise KeyError(f"Could not find key containing '{keyword}' in obs_groups['{group_name}']")

    def _get_group_steps(self, obs, group_name, keyword):
        for key in self.obs_groups.get(group_name, []):
            if keyword in key:
                return obs[key].shape[1]
        return 0

    def get_actions_log_prob(self, actions):
        dist = Normal(self.action_mean, self.action_std)
        return dist.log_prob(actions).sum(dim=-1)

    def reset(self, env_ids=None):
        pass

class TransformerEncoderMLPActorMLPCritic(nn.Module):
    is_recurrent: bool = False

    def __init__(
        self, 
        obs: dict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        d_model: int, 
        nhead: int,
        num_encoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        init_noise_std: float,
        mlp_hidden_dims: list[int], 
        activation: str,
        actor_proprio_normalization: bool,
        critic_proprio_normalization: bool,
        future_motion_normalization: bool,
        **kwargs,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.obs_groups = obs_groups
        self.d_model = d_model
        
        self.actor_proprio_normalization = actor_proprio_normalization
        self.critic_proprio_normalization = critic_proprio_normalization
        self.future_motion_normalization = future_motion_normalization

        self.actor_proprio_dim = self._get_group_dim(obs, "policy", "proprio_history")
        self.actor_future_dim = self._get_group_dim(obs, "policy", "future_motion")
        self.critic_proprio_dim = self._get_group_dim(obs, "critic", "proprio_history")
        
        self.future_steps = self._get_group_steps(obs, "policy", "future_motion")
        self.history_steps = self._get_group_steps(obs, "policy", "proprio_history")
        if actor_proprio_normalization:
            self.actor_proprio_norm = EmpiricalNormalization((self.actor_proprio_dim,))
        else:
            self.actor_proprio_norm = torch.nn.Identity()
        if critic_proprio_normalization:
            self.critic_proprio_norm = EmpiricalNormalization((self.critic_proprio_dim,))
        else:
            self.critic_proprio_norm = torch.nn.Identity()
        if future_motion_normalization:
            self.motion_future_norm = EmpiricalNormalization((self.actor_future_dim,))
        else:
            self.motion_future_norm = torch.nn.Identity()
        self.future_embedding_mlp = nn.Sequential(
            nn.Linear(self.actor_future_dim, d_model),
            nn.ELU(),
            nn.Linear(d_model, d_model)
        )
        self.pos_encoder = PositionalEncoding(d_model, max_len=self.future_steps)
        actor_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.actor_encoder = nn.TransformerEncoder(actor_layer, num_layers=num_encoder_layers)
        self.future_output_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ELU(),
            nn.Linear(d_model, d_model)
        )
        actor_mlp_input_dim = d_model + (self.actor_proprio_dim * self.history_steps)

        self.actor_mlp = self._build_mlp(actor_mlp_input_dim, mlp_hidden_dims, num_actions, activation)
        self.critic_mlp = self._build_mlp(self.critic_proprio_dim, mlp_hidden_dims, 1, activation)
        
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.apply(self._init_weights)
        
        if isinstance(self.actor_mlp[-1], nn.Linear):
            nn.init.constant_(self.actor_mlp[-1].weight, 1e-3)
            nn.init.constant_(self.actor_mlp[-1].bias, 0.0)

    def _build_mlp(self, input_dim, hidden_dims, output_dim, activation_name):
        layers = []
        curr_dim = input_dim
        act_class = getattr(nn, activation_name.upper()) if isinstance(activation_name, str) else nn.ELU
        for h in hidden_dims:
            layers.append(nn.Linear(curr_dim, h))
            layers.append(act_class())
            curr_dim = h
        layers.append(nn.Linear(curr_dim, output_dim))
        return nn.Sequential(*layers)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def _forward_actor_encoder(self, future):
        x = self.future_embedding_mlp(future)
        x = self.pos_encoder(x)
        x = self.actor_encoder(x)
        first_token = x[:, 0, :]
        future_latent = self.future_output_mlp(first_token)
        return future_latent

    def act(self, observations: dict, **kwargs):
        future = self.motion_future_norm(observations["future_motion"])
        proprio = self.actor_proprio_norm(observations["policy_proprio_history"])
        future_latent = self._forward_actor_encoder(future)
        proprio = proprio.view(proprio.shape[0], -1)
        combined_input = torch.cat([future_latent, proprio], dim=-1)
        
        self.action_mean = self.actor_mlp(combined_input)
        self.action_std = self.std.expand_as(self.action_mean)
        
        dist = Normal(self.action_mean, self.action_std)
        if self.training:
            self.entropy = dist.entropy().sum(dim=-1)
        return dist.sample()

    def get_value(self, observations: dict) -> torch.Tensor:
        proprio = observations["critic_proprio_history"][:, -1, :]
        proprio = self.critic_proprio_norm(proprio)
        return self.critic_mlp(proprio)

    def evaluate(self, observations: dict, **kwargs) -> torch.Tensor:
        return self.get_value(observations)

    def act_inference(self, observations: dict):
        with torch.no_grad():
            future = self.motion_future_norm(observations["future_motion"])
            proprio = self.actor_proprio_norm(observations["policy_proprio_history"])
            future_latent = self._forward_actor_encoder(future)
            proprio = proprio.view(proprio.shape[0], -1)
            combined_input = torch.cat([future_latent, proprio], dim=-1)
            return self.actor_mlp(combined_input)

    def update_normalization(self, obs: dict) -> None:
        if self.future_motion_normalization and "future_motion" in obs:
            self.motion_future_norm.update(obs["future_motion"].view(-1, self.actor_future_dim))
        if self.actor_proprio_normalization and "policy_proprio_history" in obs:
            self.actor_proprio_norm.update(obs["policy_proprio_history"].view(-1, self.actor_proprio_dim))
        if self.critic_proprio_normalization and "critic_proprio_history" in obs:
            self.critic_proprio_norm.update(obs["critic_proprio_history"].view(-1, self.critic_proprio_dim))

    def _get_group_dim(self, obs, group_name, keyword):
        for key in self.obs_groups.get(group_name, []):
            if keyword in key:
                return obs[key].shape[-1]
        raise KeyError(f"Could not find key containing '{keyword}' in obs_groups['{group_name}']")

    def _get_group_steps(self, obs, group_name, keyword):
        for key in self.obs_groups.get(group_name, []):
            if keyword in key:
                return obs[key].shape[1]
        return 0

    def get_actions_log_prob(self, actions):
        dist = Normal(self.action_mean, self.action_std)
        return dist.log_prob(actions).sum(dim=-1)

    def reset(self, env_ids=None):
        pass

class TransformerEncoderActorMLPCritic(nn.Module):
    is_recurrent: bool = False

    def __init__(
        self, 
        obs: dict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        d_model: int, 
        nhead: int,
        num_encoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        init_noise_std: float,
        mlp_hidden_dims: list[int], 
        activation: str = "ELU",
        **kwargs,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.obs_groups = obs_groups
        self.d_model = d_model

        self.actor_proprio_dim = self._get_group_dim(obs, "policy", "proprio_history")
        self.critic_proprio_dim = self._get_group_dim(obs, "critic", "proprio_history")
        self.actor_future_dim = self._get_group_dim(obs, "policy", "future_motion")
        self.future_steps = self._get_group_steps(obs, "policy", "future_motion")
        self.history_steps = self._get_group_steps(obs, "policy", "proprio_history")

        self.actor_proprio_norm = EmpiricalNormalization((self.actor_proprio_dim,))
        self.critic_proprio_norm = EmpiricalNormalization((self.critic_proprio_dim,))

        self.pos_encoder = PositionalEncoding(d_model, max_len=self.history_steps + self.future_steps)

        self.actor_proprio_proj = nn.Linear(self.actor_proprio_dim, d_model)
        self.actor_future_proj = nn.Linear(self.actor_proprio_dim, d_model)
        actor_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.actor_encoder = nn.TransformerEncoder(actor_layer, num_layers=num_encoder_layers)
        self.actor_mlp = self._build_mlp(d_model, mlp_hidden_dims, num_actions, activation)
        self.critic_mlp = self._build_mlp(self.critic_proprio_dim, mlp_hidden_dims, 1, activation)
        
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.apply(self._init_weights)
        
        if isinstance(self.actor_mlp[-1], nn.Linear):
            nn.init.constant_(self.actor_mlp[-1].weight, 1e-3)
            nn.init.constant_(self.actor_mlp[-1].bias, 0.0)

    def _build_mlp(self, input_dim, hidden_dims, output_dim, activation_name):
        layers = []
        curr_dim = input_dim
        act_class = getattr(nn, activation_name.upper()) if isinstance(activation_name, str) else nn.ELU
        for h in hidden_dims:
            layers.append(nn.Linear(curr_dim, h))
            layers.append(act_class())
            curr_dim = h
        layers.append(nn.Linear(curr_dim, output_dim))
        return nn.Sequential(*layers)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def _forward_transformer(self, proprio, future):
        p_emb = self.actor_proprio_proj(proprio)
        f_emb = self.actor_future_proj(future)
        seq = torch.cat([p_emb, f_emb], dim=1)
        seq = self.pos_encoder(seq)
        out = self.actor_encoder(seq)
        return out[:, self.history_steps - 1, :]

    def act(self, observations: dict, **kwargs):
        proprio = self.actor_proprio_norm(observations["policy_proprio_history"])
        future = self.actor_proprio_norm(observations["future_motion"])
        
        latent = self._forward_transformer(proprio, future)
        self.action_mean = self.actor_mlp(latent)
        self.action_std = self.std.expand_as(self.action_mean)
        
        dist = Normal(self.action_mean, self.action_std)
        if self.training:
            self.entropy = dist.entropy().sum(dim=-1)
        return dist.sample()

    def get_value(self, observations: dict) -> torch.Tensor:
        proprio = observations["critic_proprio_history"][:, -1, :]
        proprio = self.critic_proprio_norm(proprio)
        return self.critic_mlp(proprio)

    def evaluate(self, observations: dict, **kwargs) -> torch.Tensor:
        return self.get_value(observations)

    def act_inference(self, observations: dict):
        with torch.no_grad():
            future = self.motion_future_norm(observations["future_motion"])
            proprio = self.actor_proprio_norm(observations["policy_proprio_history"])
            latent = self._forward_transformer(proprio, future, self.actor_proprio_proj, self.actor_future_proj, self.actor_encoder)
            return self.actor_mlp(latent)

    def update_normalization(self, obs: dict) -> None:
        if "policy_proprio_history" in obs:
            self.actor_proprio_norm.update(obs["policy_proprio_history"].view(-1, self.actor_proprio_dim))
        if "critic_proprio_history" in obs:
            self.critic_proprio_norm.update(obs["critic_proprio_history"].view(-1, self.critic_proprio_dim))

    def _get_group_dim(self, obs, group_name, keyword):
        for key in self.obs_groups.get(group_name, []):
            if keyword in key:
                return obs[key].shape[-1]
        raise KeyError(f"Could not find key containing '{keyword}' in obs_groups['{group_name}']")

    def _get_group_steps(self, obs, group_name, keyword):
        for key in self.obs_groups.get(group_name, []):
            if keyword in key:
                return obs[key].shape[1]
        return 0

    def get_actions_log_prob(self, actions):
        dist = Normal(self.action_mean, self.action_std)
        return dist.log_prob(actions).sum(dim=-1)

    def reset(self, env_ids=None):
        pass

class MOEMLPActorMLPCritic(nn.Module):
    is_recurrent: bool = False

    def __init__(
        self, 
        obs: dict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        num_experts: int,
        mlp_hidden_dims: list[int], 
        init_noise_std: float,
        activation: str,
        **kwargs,
    ):
        super().__init__()
        self.num_actions = num_actions
        print("-" * 100)
        print(self.num_actions)
        self.obs_groups = obs_groups
        self.num_experts = num_experts

        self.actor_proprio_dim = self._get_group_dim(obs, "policy", "proprio_history")
        self.critic_proprio_dim = self._get_group_dim(obs, "critic", "proprio_history")

        self.actor_proprio_norm = EmpiricalNormalization((self.actor_proprio_dim,))
        self.critic_proprio_norm = EmpiricalNormalization((self.critic_proprio_dim,))

        self.gate_mlp = self._build_mlp(
            self.actor_proprio_dim, 
            mlp_hidden_dims, 
            num_experts, 
            activation
        )

        self.experts = nn.ModuleList([
            self._build_mlp(self.actor_proprio_dim, mlp_hidden_dims, num_actions, activation)
            for _ in range(num_experts)
        ])

        self.critic_mlp = self._build_mlp(self.critic_proprio_dim, mlp_hidden_dims, 1, activation)
        
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.apply(self._init_weights)

    def _get_proprio(self, observations: dict, key: str, norm_layer: nn.Module):
        proprio = observations[key][:, -1, :]
        return norm_layer(proprio)

    def act(self, observations: dict, **kwargs):
        proprio = self._get_proprio(observations, "policy_proprio_history", self.actor_proprio_norm)
        gate_weights = F.softmax(self.gate_mlp(proprio), dim=-1)
        expert_outs = torch.stack([exp(proprio) for exp in self.experts], dim=1)
        self.action_mean = torch.bmm(gate_weights.unsqueeze(1), expert_outs).squeeze(1)
        self.action_std = self.std.expand_as(self.action_mean)
        
        dist = Normal(self.action_mean, self.action_std)
        if self.training:
            self.entropy = dist.entropy().sum(dim=-1)
        return dist.sample()

    def act_inference(self, observations: dict):
        with torch.no_grad():
            proprio = self._get_proprio(observations, "policy_proprio_history", self.actor_proprio_norm)
            gate_weights = F.softmax(self.gate_mlp(proprio), dim=-1)
            expert_outs = torch.stack([exp(proprio) for exp in self.experts], dim=1)
            return torch.bmm(gate_weights.unsqueeze(1), expert_outs).squeeze(1)

    def get_value(self, observations: dict) -> torch.Tensor:
        proprio = self._get_proprio(observations, "critic_proprio_history", self.critic_proprio_norm)
        return self.critic_mlp(proprio)

    def evaluate(self, observations: dict, **kwargs) -> torch.Tensor:
        return self.get_value(observations)

    def _build_mlp(self, input_dim, hidden_dims, output_dim, activation_name):
        layers = []
        curr_dim = input_dim
        act_class = getattr(nn, activation_name.upper()) if isinstance(activation_name, str) else nn.ELU
        for h in hidden_dims:
            layers.append(nn.Linear(curr_dim, h))
            layers.append(act_class())
            curr_dim = h
        layers.append(nn.Linear(curr_dim, output_dim))
        return nn.Sequential(*layers)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def update_normalization(self, obs: dict) -> None:
        if "policy_proprio_history" in obs:
            self.actor_proprio_norm.update(obs["policy_proprio_history"][:, -1, :])
        if "critic_proprio_history" in obs:
            self.critic_proprio_norm.update(obs["critic_proprio_history"][:, -1, :])

    def _get_group_dim(self, obs, group_name, keyword):
        for key in self.obs_groups.get(group_name, []):
            if keyword in key:
                return obs[key].shape[-1]
        raise KeyError(f"Could not find key containing '{keyword}' in obs_groups['{group_name}']")

    def get_actions_log_prob(self, actions):
        dist = Normal(self.action_mean, self.action_std)
        return dist.log_prob(actions).sum(dim=-1)

    def reset(self, env_ids=None):
        pass

class MOEMLPTransformerEncoderActorMLPCritic(nn.Module):
    is_recurrent: bool = False

    def __init__(
        self, 
        obs: dict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        d_model: int, 
        nhead: int,
        num_encoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        num_experts: int,
        mlp_hidden_dims: list[int], 
        init_noise_std: float,
        activation: str,
        **kwargs,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.obs_groups = obs_groups
        self.num_experts = num_experts

        self.actor_proprio_dim = self._get_group_dim(obs, "policy", "proprio_history")
        self.critic_proprio_dim = self._get_group_dim(obs, "critic", "proprio_history")
        self.actor_future_dim = self._get_group_dim(obs, "policy", "future_motion")
        self.future_steps = self._get_group_steps(obs, "policy", "future_motion")
        self.history_steps = self._get_group_steps(obs, "policy", "proprio_history")

        self.actor_future_norm = EmpiricalNormalization((self.actor_future_dim,))
        self.actor_proprio_norm = EmpiricalNormalization((self.actor_proprio_dim,))
        self.critic_proprio_norm = EmpiricalNormalization((self.critic_proprio_dim,))

        self.future_proj = nn.Linear(self.actor_future_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=self.future_steps)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.gate_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        gate_hidden_dim = d_model // 2
        self.gate_head = nn.Sequential(
            nn.Linear(d_model, gate_hidden_dim),
            nn.ELU(),
            nn.Linear(gate_hidden_dim, num_experts)
        )

        self.experts = nn.ModuleList([
            self._build_mlp(self.actor_proprio_dim, mlp_hidden_dims, num_actions, activation)
            for _ in range(num_experts)
        ])

        self.critic_mlp = self._build_mlp(self.critic_proprio_dim, mlp_hidden_dims, 1, activation)
        
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        # self.apply(self._init_weights)

    def _forward_gate(self, future_motion):
        x = self.future_proj(future_motion)
        x = self.pos_encoder(x)
        latent = self.gate_transformer(x)
        gate_feature = torch.mean(latent, dim=1) 
        gate_weights = F.softmax(self.gate_head(gate_feature), dim=-1)
        return gate_weights

    def act(self, observations: dict, **kwargs):
        future_motion = self.actor_future_norm(observations["future_motion"])
        policy_proprio_history = self.actor_proprio_norm(observations["policy_proprio_history"])
        gate_weights = self._forward_gate(future_motion) # [B, num_experts]
        policy_proprio = policy_proprio_history[:, -1, :]
        expert_outs = torch.stack([exp(policy_proprio) for exp in self.experts], dim=1) # [B, num_experts, num_actions]
        
        self.action_mean = torch.bmm(gate_weights.unsqueeze(1), expert_outs).squeeze(1)
        self.action_std = self.std.expand_as(self.action_mean)
        
        dist = Normal(self.action_mean, self.action_std)
        if self.training:
            self.entropy = dist.entropy().sum(dim=-1)
        return dist.sample()

    def act_inference(self, observations: dict):
        with torch.no_grad():
            future_motion = self.actor_future_norm(observations["future_motion"])
            proprio_history = self.actor_proprio_norm(observations["policy_proprio_history"])
            
            gate_weights = self._forward_gate(future_motion)
            current_proprio = proprio_history[:, -1, :]
            
            expert_outs = torch.stack([exp(current_proprio) for exp in self.experts], dim=1)
            return torch.bmm(gate_weights.unsqueeze(1), expert_outs).squeeze(1)

    def evaluate(self, observations: dict, **kwargs) -> torch.Tensor:
        return self.get_value(observations)

    def get_value(self, observations: dict) -> torch.Tensor:
        proprio = observations["critic_proprio_history"][:, -1, :]
        proprio = self.critic_proprio_norm(proprio)
        return self.critic_mlp(proprio)

    def _build_mlp(self, input_dim, hidden_dims, output_dim, activation_name):
        layers = []
        curr_dim = input_dim
        act_class = getattr(nn, activation_name.upper()) if isinstance(activation_name, str) else nn.ELU
        for h in hidden_dims:
            layers.append(nn.Linear(curr_dim, h)); layers.append(act_class())
            curr_dim = h
        layers.append(nn.Linear(curr_dim, output_dim))
        return nn.Sequential(*layers)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def update_normalization(self, obs: dict) -> None:
        if "future_motion" in obs:
            self.actor_future_norm.update(obs["future_motion"].reshape(-1, self.actor_future_dim))
        if "policy_proprio_history" in obs:
            self.actor_proprio_norm.update(obs["policy_proprio_history"].reshape(-1, self.actor_proprio_dim))
        if "critic_proprio_history" in obs:
            self.critic_proprio_norm.update(obs["critic_proprio_history"].reshape(-1, self.critic_proprio_dim))

    def _get_group_dim(self, obs, group_name, keyword):
        for key in self.obs_groups.get(group_name, []):
            if keyword in key:
                return obs[key].shape[-1]
        raise KeyError(f"Could not find key containing '{keyword}' in obs_groups['{group_name}']")

    def _get_group_steps(self, obs, group_name, keyword):
        for key in self.obs_groups.get(group_name, []):
            if keyword in key:
                return obs[key].shape[1]
        return 0

    def get_actions_log_prob(self, actions):
        dist = Normal(self.action_mean, self.action_std)
        return dist.log_prob(actions).sum(dim=-1)

    def reset(self, env_ids=None):
        pass


class MLPActorMLPCritic(nn.Module):
    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool,
        critic_obs_normalization: bool,
        actor_hidden_dims: tuple[int] | list[int],
        critic_hidden_dims: tuple[int] | list[int],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.obs_groups = obs_groups

        self.actor_proprio_dim = self._get_group_dim(obs, "policy", "proprio_history")
        self.critic_proprio_dim = self._get_group_dim(obs, "critic", "proprio_history")
        self.history_steps = self._get_group_steps(obs, "policy", "proprio_history")

        self.actor = self._build_mlp(self.actor_proprio_dim, actor_hidden_dims, num_actions, activation)
        print(f"Actor MLP: {self.actor}")

        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization((self.actor_proprio_dim,))
        else:
            self.actor_obs_normalizer = torch.nn.Identity()

        self.critic = self._build_mlp(self.critic_proprio_dim, critic_hidden_dims, 1, activation)
        print(f"Critic MLP: {self.critic}")

        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization((self.critic_proprio_dim,))
        else:
            self.critic_obs_normalizer = torch.nn.Identity()

        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))

        self.distribution = None
        # self.apply(self._init_weights)

        Normal.set_default_validate_args(False)

    def _update_distribution(self, obs: TensorDict) -> None:
        mean = self.actor(obs)
        std = self.std.expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, obs: TensorDict, **kwargs) -> torch.Tensor:
        actor_obs = obs["policy_proprio_history"][:, -1, :] 
        actor_obs = self.actor_obs_normalizer(actor_obs)
        
        self.action_mean = self.actor(actor_obs)
        self.action_std = self.std.expand_as(self.action_mean)
        self.distribution = Normal(self.action_mean, self.action_std)
        
        if self.training:
            self.entropy = self.distribution.entropy().sum(dim=-1)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        actor_obs = obs["policy_proprio_history"][:, -1, :]
        actor_obs = self.actor_obs_normalizer(actor_obs)
        return self.actor(actor_obs)

    def evaluate(self, obs: TensorDict, **kwargs) -> torch.Tensor:
        critic_obs = obs["critic_proprio_history"][:, -1, :]
        critic_obs = self.critic_obs_normalizer(critic_obs)
        return self.critic(critic_obs)

    def _get_group_dim(self, obs, group_name, keyword):
        for key in self.obs_groups.get(group_name, []):
            if keyword in key:
                return obs[key].shape[-1]
        raise KeyError(f"Could not find key containing '{keyword}' in obs_groups['{group_name}']")

    def _get_group_steps(self, obs, group_name, keyword):
        for key in self.obs_groups.get(group_name, []):
            if keyword in key:
                return obs[key].shape[1]
        return 0

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def reset(self, env_ids=None):
        pass

    def _build_mlp(self, input_dim, hidden_dims, output_dim, activation_name):
        layers = []
        curr_dim = input_dim
        act_class = getattr(nn, activation_name.upper()) if isinstance(activation_name, str) else nn.ELU
        for h in hidden_dims:
            layers.append(nn.Linear(curr_dim, h))
            layers.append(act_class())
            curr_dim = h
        layers.append(nn.Linear(curr_dim, output_dim))
        return nn.Sequential(*layers)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.actor_obs_normalization:
            self.actor_obs_normalizer.update(obs["policy_proprio_history"][:, -1, :])
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(obs["critic_proprio_history"][:, -1, :])

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        super().load_state_dict(state_dict, strict=strict)
        return True