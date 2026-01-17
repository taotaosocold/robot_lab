import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import traceback
import math

# --- 1. 解决路径问题 ---
MODEL_DIR = "/home/ubuntu/Desktop/robot_lab/source/robot_lab/robot_lab/tasks/manager_based/MotionTracking/config/g1/agents"
if MODEL_DIR not in sys.path:
    sys.path.append(MODEL_DIR)

from Transformer_ActorCritic import TransformerEncoderMLPActorCritic

class CleanMultiheadAttention(nn.Module):
    def __init__(self, original_mha):
        super().__init__()
        self.embed_dim = original_mha.embed_dim
        self.num_heads = original_mha.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        
        self.batch_first = getattr(original_mha, "batch_first", True) 
        self.dropout = original_mha.dropout # 某些版本可能还会检查这个
        
        assert self.head_dim * self.num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        
        if original_mha.in_proj_weight is not None:
            q_w, k_w, v_w = original_mha.in_proj_weight.chunk(3, dim=0)
            self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
            self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
            self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
            self.q_proj.weight.data.copy_(q_w)
            self.k_proj.weight.data.copy_(k_w)
            self.v_proj.weight.data.copy_(v_w)
            
            if original_mha.in_proj_bias is not None:
                q_b, k_b, v_b = original_mha.in_proj_bias.chunk(3, dim=0)
                self.q_proj.bias.data.copy_(q_b)
                self.k_proj.bias.data.copy_(k_b)
                self.v_proj.bias.data.copy_(v_b)
        
        self.out_proj = original_mha.out_proj

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None, need_weights=False, is_causal=False):
        bsz, tgt_len, _ = query.size()
        
        q = self.q_proj(query).view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask
            
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_probs, v)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        
        return attn_output, None

class StandardTransformerLayer(nn.Module):
    def __init__(self, original_layer):
        super().__init__()
        self.self_attn = CleanMultiheadAttention(original_layer.self_attn)
        self.linear1 = original_layer.linear1
        self.linear2 = original_layer.linear2
        self.norm1 = original_layer.norm1
        self.norm2 = original_layer.norm2
        self.dropout = nn.Dropout(original_layer.dropout.p)
        self.dropout1 = nn.Dropout(original_layer.dropout1.p)
        self.dropout2 = nn.Dropout(original_layer.dropout2.p)
        self.activation = F.elu # 你的模型使用的是 ELU

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        x = src
        norm_x = self.norm1(x)
        attn_out, _ = self.self_attn(norm_x, norm_x, norm_x, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        x = x + self.dropout1(attn_out)
        norm_x = self.norm2(x)
        ff_out = self.linear2(self.dropout(self.activation(self.linear1(norm_x))))
        x = x + self.dropout2(ff_out)
        return x

class ActorWrapper(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.model = original_model
    def forward(self, proprio_history, future_motion):
        obs = {"policy_proprio_history": proprio_history, "future_motion": future_motion}
        return self.model.act_inference(obs)

def export_all(checkpoint_path):
    device = "cpu"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    export_path = os.path.join(checkpoint_dir, "exported")
    os.makedirs(export_path, exist_ok=True)
    onnx_file = os.path.join(export_path, "policy.onnx")
    jit_file = os.path.join(export_path, "policy.pt")

    ACTOR_PROPRIO_DIM, CRITIC_PROPRIO_DIM, FUTURE_DIM = 259, 323, 256
    FUTURE_STEPS, HISTORY_STEPS, NUM_ACTIONS = 35, 6, 23

    model = TransformerEncoderMLPActorCritic(
        obs={"future_motion": torch.zeros(1, 35, 256), "policy_proprio_history": torch.zeros(1, 6, 259), "critic_proprio_history": torch.zeros(1, 6, 323)},
        obs_groups={"policy": ["future_motion", "policy_proprio_history"], "critic": ["future_motion", "critic_proprio_history"]},
        num_actions=NUM_ACTIONS, d_model=256, nhead=2, num_encoder_layers=1, dim_feedforward=256,
        dropout=0.0, init_noise_std=1.0, mlp_hidden_dims=[1024, 512, 256, 128], activation="ELU"
    ).to(device)

    print(f"[INFO] 正在加载权重: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict.get('model_state_dict', state_dict))
    model.eval()

    def replace_layers(module):
        for name, child in module.named_children():
            if isinstance(child, nn.TransformerEncoderLayer):
                setattr(module, name, StandardTransformerLayer(child))
            else:
                replace_layers(child)
    
    replace_layers(model)
    print("[INFO] 已完成注意力模块与 Transformer 层全手动拆解补丁")

    wrapper = ActorWrapper(model)
    test_proprio = torch.randn(1, HISTORY_STEPS, ACTOR_PROPRIO_DIM)
    test_future = torch.randn(1, FUTURE_STEPS, FUTURE_DIM)

    print(f"[INFO] 正在导出 ONNX 至: {onnx_file}")
    try:
        torch.onnx.export(
            wrapper, (test_proprio, test_future), onnx_file,
            opset_version=15, do_constant_folding=True,
            input_names=["proprio_history", "future_motion"], output_names=["actions"],
            dynamic_axes={"proprio_history": {0: "batch_size"}, "future_motion": {0: "batch_size"}, "actions": {0: "batch_size"}}
        )
        print("[SUCCESS] ONNX 导出成功！")
    except Exception:
        traceback.print_exc()

    print(f"[INFO] 正在导出 JIT 至: {jit_file}")
    try:
        with torch.no_grad():
            traced_model = torch.jit.trace(wrapper, (test_proprio, test_future))
            traced_model.save(jit_file)
        print("[SUCCESS] JIT 导出成功！")
    except Exception:
        traceback.print_exc()

if __name__ == "__main__":
    ckpt = "/home/ubuntu/Desktop/robot_lab/logs/rsl_rl/unitree_g1_MotionTracking_flat/2026-01-06_20-11-18/model_6300.pt"
    export_all(ckpt)