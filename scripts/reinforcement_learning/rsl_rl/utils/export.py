import torch
import torch.nn as nn
import os

def export_policy_as_onnx(model, path, filename="policy.onnx"):
    class OnnxWrapper(nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.policy = policy

        def forward(self, proprio_history, future_motion):
            obs_dict = {
                "policy_proprio_history": proprio_history,
                "future_motion": future_motion
            }
            return self.policy.act_inference(obs_dict)

    os.makedirs(path, exist_ok=True)
    onnx_file = os.path.join(path, filename)
    model.eval().to("cpu")
    wrapper = OnnxWrapper(model)

    from torch.backends.cuda import sdp_kernel

    history_len = getattr(model, "history_steps", 6)
    future_steps = getattr(model, "future_steps", 35)
    proprio_dim = model.actor_proprio_dim
    future_dim = model.actor_future_dim

    dummy_proprio = torch.randn(1, history_len, proprio_dim)
    dummy_future = torch.randn(1, future_steps, future_dim)

    print(f"[INFO] 正在导出 ONNX 模型至: {onnx_file}")
    try:
            with sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
                torch.onnx.export(
                    wrapper,
                    (dummy_proprio, dummy_future),
                    onnx_file,
                    input_names=["proprio_history", "future_motion"],
                    output_names=["actions"],
                    dynamic_axes={
                        "proprio_history": {0: "batch_size"},
                        "future_motion": {0: "batch_size"},
                        "actions": {0: "batch_size"}
                    },
                    opset_version=17,
                    do_constant_folding=True
                )
        print(f"[INFO] ONNX 导出成功！")
    except Exception as e:
        print(f"[ERROR] ONNX 导出依然失败: {e}")


def export_policy_as_jit(model, path, filename="policy.pt"):
    class JitWrapper(nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.policy = policy

        def forward(self, proprio_history, future_motion):
            obs_dict = {
                "policy_proprio_history": proprio_history,
                "future_motion": future_motion
            }
            return self.policy.act_inference(obs_dict)

    os.makedirs(path, exist_ok=True)
    jit_file = os.path.join(path, filename)
    
    orig_device = next(model.parameters()).device
    
    model.eval().to("cpu")
    wrapper = JitWrapper(model)

    # 获取维度
    history_len = getattr(model, "history_steps", 6)
    future_steps = getattr(model, "future_steps", 35)
    proprio_dim = model.actor_proprio_dim
    future_dim = model.actor_future_dim

    dummy_proprio = torch.randn(1, history_len, proprio_dim)
    dummy_future = torch.randn(1, future_steps, future_dim)

    print(f"[INFO] 正在导出 JIT 模型 (TorchScript) 至: {jit_file}")
    try:
        with torch.no_grad():
            traced_model = torch.jit.trace(wrapper, (dummy_proprio, dummy_future))
            traced_model.save(jit_file)
        print(f"[INFO] JIT 导出成功！")
    finally:
        model.to(orig_device)