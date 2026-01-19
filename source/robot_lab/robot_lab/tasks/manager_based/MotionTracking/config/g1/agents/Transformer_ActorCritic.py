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

class TransformerEncoderMLPActorCritic(nn.Module):
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

        self.pos_encoder = PositionalEncoding(d_model, max_len=self.future_steps)
        self.actor_future_proj = nn.Linear(self.actor_future_dim, d_model)  
        actor_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.actor_future_encoder = nn.TransformerEncoder(actor_layer, num_layers=num_encoder_layers)

        self.critic_future_proj = nn.Linear(self.critic_future_dim, d_model)
        critic_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.critic_future_encoder = nn.TransformerEncoder(critic_layer, num_layers=num_encoder_layers)

        actor_mlp_input_dim = d_model + (self.history_steps * self.actor_proprio_dim)
        critic_mlp_input_dim = d_model + (self.history_steps * self.critic_proprio_dim)

        self.actor_mlp = self._build_mlp(actor_mlp_input_dim, mlp_hidden_dims, num_actions, activation)
        self.critic_mlp = self._build_mlp(critic_mlp_input_dim, mlp_hidden_dims, 1, activation)
        
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

    def _encode_future(self, future, proj_layer, encoder):
        x = proj_layer(future)
        x = self.pos_encoder(x)
        x = encoder(x)
        latent = torch.mean(x, dim=1) 
        
        return latent

    def act(self, observations: dict, **kwargs):
        future = self.motion_future_norm(observations["future_motion"])
        proprio = self.actor_proprio_norm(observations["policy_proprio_history"])
        future_latent = self._encode_future(future, self.actor_future_proj, self.actor_future_encoder)
        
        batch_size = proprio.shape[0]
        proprio_flat = proprio.view(batch_size, -1) 
        
        combined_input = torch.cat([future_latent, proprio_flat], dim=-1)
        
        self.action_mean = self.actor_mlp(combined_input)
        self.action_std = self.std.expand_as(self.action_mean)
        
        dist = Normal(self.action_mean, self.action_std)
        if self.training:
            self.entropy = dist.entropy().sum(dim=-1)
        return dist.sample()

    def get_value(self, observations: dict) -> torch.Tensor:
        future = self.motion_future_norm(observations["future_motion"])
        proprio = self.critic_proprio_norm(observations["critic_proprio_history"])
        
        future_latent = self._encode_future(future, self.critic_future_proj, self.critic_future_encoder)
        proprio_flat = proprio.view(proprio.shape[0], -1)
        
        combined_input = torch.cat([future_latent, proprio_flat], dim=-1)
        
        return self.critic_mlp(combined_input)

    def evaluate(self, observations: dict, **kwargs) -> torch.Tensor:
        return self.get_value(observations)

    def act_inference(self, observations: dict):
        with torch.no_grad():
            future = self.motion_future_norm(observations["future_motion"])
            proprio = self.actor_proprio_norm(observations["policy_proprio_history"])
            
            future_latent = self._encode_future(future, self.actor_future_proj, self.actor_future_encoder)
            proprio_flat = proprio.view(proprio.shape[0], -1)
            
            combined_input = torch.cat([future_latent, proprio_flat], dim=-1)
            return self.actor_mlp(combined_input)

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