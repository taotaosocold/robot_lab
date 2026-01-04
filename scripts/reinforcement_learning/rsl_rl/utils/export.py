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

    history_len = getattr(model, "history_steps", 6)
    future_steps = getattr(model, "future_steps", 35)
    proprio_dim = model.actor_proprio_dim
    future_dim = model.actor_future_dim

    dummy_proprio = torch.randn(1, history_len, proprio_dim)
    dummy_future = torch.randn(1, future_steps, future_dim)

    print(f"[INFO] 正在导出模型至: {onnx_file}")
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
        opset_version=14,
        do_constant_folding=True
    )
    print(f"[INFO] ONNX 导出成功！")