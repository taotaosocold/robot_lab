import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import math
import traceback
import os
import sys

# 1. 显式添加模型文件所在的绝对路径
MODEL_PATH = "/home/ubuntu/Desktop/robot_lab/source/robot_lab/robot_lab/tasks/manager_based/MotionTracking/config/g1/agents"
if MODEL_PATH not in sys.path:
    sys.path.append(MODEL_PATH)

# 2. 现在再尝试导入
try:
    from Transformer_ActorCritic import MOEMLPTransformerEncoderActorMLPCritic, MOEMLPTransformerEncoderActorCritic
except ImportError as e:
    print(f"Error: 无法找到 Transformer_ActorCritic。请检查路径: {MODEL_PATH}")
    raise e
# ==========================================
# 1. 基础组件补丁 (解决 Transformer 导出 ONNX 报错)
# ==========================================
class CleanMultiheadAttention(nn.Module):
    def __init__(self, original_mha):
        super().__init__()
        self.embed_dim = original_mha.embed_dim
        self.num_heads = original_mha.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        
        self.batch_first = getattr(original_mha, "batch_first", True)
        # 权重提取
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
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, tgt_len, self.embed_dim)
        return self.out_proj(attn_output), None

class StandardTransformerLayer(nn.Module):
    def __init__(self, original_layer):
        super().__init__()
        self.self_attn = CleanMultiheadAttention(original_layer.self_attn)
        self.linear1, self.linear2 = original_layer.linear1, original_layer.linear2
        self.norm1, self.norm2 = original_layer.norm1, original_layer.norm2
        self.activation = F.gelu # 你的配置里 Gate 使用的是 gelu

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        # Norm First 架构
        x = src
        norm_x = self.norm1(x)
        attn_out, _ = self.self_attn(norm_x, norm_x, norm_x)
        x = x + attn_out
        norm_x = self.norm2(x)
        x = x + self.linear2(self.activation(self.linear1(norm_x)))
        return x

# ==========================================
# 2. 导出包装器
# ==========================================
class MOEPolicyExporterWrapper(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.actor = original_model # 关键：必须叫 actor 兼容 IsaacLab

    def forward(self, future_motion, proprio_history):
        """
        注意：输入顺序根据你的推理习惯决定
        future_motion: [B, 15, dim]
        proprio_history: [B, 1, dim]
        """
        obs = {
            "future_motion": future_motion,
            "policy_proprio_history": proprio_history
        }
        # 直接调用你的推理接口
        return self.actor.act_inference(obs)

# ==========================================
# 3. 主导出函数
# ==========================================
def export_moe_model(checkpoint_path, save_dir):
    # 根据你的 Config 定义维度
    # future_steps = 15, history_length = 1
    # 请确保这里的维度与训练时的 obs_group 拼接后的维度完全一致
    FUTURE_DIM = 256  # 示例值，需根据 ObsTerm 实际输出维度计算
    PROPRIO_DIM = 334 # 示例值
    NUM_ACTIONS = 23
    NUM_EXPERTS = 15   # 示例值
    
    device = torch.device("cpu")
    os.makedirs(save_dir, exist_ok=True)

    # 导入你的模型类
    from Transformer_ActorCritic import MOEMLPTransformerEncoderActorMLPCritic, MOEMLPTransformerEncoderActorCritic

    # 初始化模型结构 (必须与训练时参数一致)
    model = MOEMLPTransformerEncoderActorCritic(
        obs={
            "future_motion": torch.zeros(1, 20, FUTURE_DIM),
            "policy_proprio_history": torch.zeros(1, 1, PROPRIO_DIM),
            "critic_proprio_history": torch.zeros(1, 1, 334)
        },
        obs_groups={
            "policy": ["future_motion", "policy_proprio_history"],
            "critic": ["future_motion", "critic_proprio_history"]
        },
        num_actions=NUM_ACTIONS,
        d_model=256,
        nhead=4,
        num_encoder_layers=1,
        dim_feedforward=256,
        dropout=0.0,
        num_experts=NUM_EXPERTS,
        mlp_hidden_dims=[512, 256, 128],
        init_noise_std=1.0,
        activation="ELU"
    ).to(device)

    # 加载权重
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict.get('model_state_dict', state_dict))
    model.eval()

    # 应用 Transformer 补丁
    def apply_patch(module):
        for name, child in module.named_children():
            if isinstance(child, nn.TransformerEncoderLayer):
                setattr(module, name, StandardTransformerLayer(child))
            else: apply_patch(child)
    apply_patch(model)

    # 封装
    wrapper = MOEPolicyExporterWrapper(model)
    
    # 模拟输入
    dummy_future = torch.randn(1, 20, FUTURE_DIM)
    dummy_proprio = torch.randn(1, 1, PROPRIO_DIM)

    # 导出 JIT
    jit_path = os.path.join(save_dir, "policy.pt")
    try:
        with torch.no_grad():
            traced_script = torch.jit.trace(wrapper, (dummy_future, dummy_proprio))
            traced_script.save(jit_path)
        print(f"[SUCCESS] JIT exported to {jit_path}")
    except Exception: traceback.print_exc()

    # 导出 ONNX
    onnx_path = os.path.join(save_dir, "policy.onnx")
    try:
        torch.onnx.export(
            wrapper, 
            (dummy_future, dummy_proprio),
            onnx_path,
            opset_version=15,
            input_names=["future_motion", "proprio_history"],
            output_names=["actions"],
            dynamic_axes={
                "future_motion": {0: "batch_size"},
                "proprio_history": {0: "batch_size"},
                "actions": {0: "batch_size"}
            }
        )
        print(f"[SUCCESS] ONNX exported to {onnx_path}")
    except Exception: traceback.print_exc()