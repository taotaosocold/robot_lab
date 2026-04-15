import enum
import math
from einops import rearrange
import numpy as np
import torch
import torch as th
from tqdm import tqdm
from .nn import sum_flat
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint
# import wandb
# from utils.dist_util import is_rank_zero


class FlowMatching:

    def __init__(
        self,
        *,
        lambda_rcxyz=0.0,
        lambda_vel=0.0,
        data_rep="rot6d",
        lambda_root_vel=0.0,
        lambda_vel_rcxyz=0.0,
        lambda_fc=0.0,
    ):
        self.data_rep = data_rep
        # xyz重建损失的权重
        self.lambda_rcxyz = lambda_rcxyz
        # 速度损失的权重
        self.lambda_vel = lambda_vel
        # rot6d旋转损失的权重
        self.lambda_root_vel = lambda_root_vel
        # xyz速度损失的权重
        self.lambda_vel_rcxyz = lambda_vel_rcxyz
        # 脚接触损失的权重
        self.lambda_fc = lambda_fc
        # 计算误差
        self.l2_loss = (
            lambda a, b: (a - b) ** 2
        )  # th.nn.MSELoss(reduction='none')  # must be None for handling mask later on.

    # 带掩码的l2误差
    def masked_l2(self, a, b, mask=None):
        # assuming a.shape == b.shape == bs, J, Jdim, seqlen
        # assuming mask.shape == bs, 1, 1, seqlen
        mask = torch.ones((a.size(0), 1, 1, a.size(1)), device=a.device)
        loss = self.l2_loss(a, b)
        loss = sum_flat(
            loss * mask.float()
        )  # gives \sigma_euclidean over unmasked elements
        n_entries = a.shape[0] * a.shape[1]
        non_zero_elements = sum_flat(mask) * n_entries
        # print('mask', mask.shape)
        # print('non_zero_elements', non_zero_elements)
        # print('loss', loss)
        mse_loss_val = loss / non_zero_elements
        # print('mse_loss_val', mse_loss_val)
        return mse_loss_val
    # 使用欧拉方法进行原始采样
    # 这里model是模型，输出是预测的速度场，z_orig是初始状态为噪声就是x_t，N为采样步数
    @torch.no_grad()
    def sample_euler_raw(self, model, z_orig, N, model_kwargs, ode_kwargs):
        # dt是欧拉步长，因为时间区间是[0,1]，所以步长为1/N
        dt = 1.0 / N
        traj = []  # to store the trajectory

        z = z_orig.detach().clone()
        bs = len(z)
        # est存储中间的估计值，比如x_1的估计值
        est = []
        return_x_est = ode_kwargs["return_x_est"]
        if return_x_est:
            return_x_est_num = ode_kwargs["return_x_est_num"]
            est_ids = [int(i * N / return_x_est_num) for i in range(return_x_est_num)]

        traj.append(z.detach().clone())
        # 循环N步
        for i in range(0, N, 1):
            t = torch.ones(bs, device=z_orig.device) * i / N
            # 这里z其实就是x_t,模型输入(x_t,t)输出预测的速度
            pred = model(z, t, **model_kwargs)
            # 这里是通过预测的速度反推x_1
            _est_now = z + (1 - i * 1.0 / N) * pred
            # 记录每一步反推得到的x_1，记录是为了后期画图去可视化模型的训练效果以及数值趋势
            est.append(_est_now.detach().clone())

            z = z.detach().clone() + pred * dt
            # 记录每一步预测的x_t
            traj.append(z.detach().clone())

        if return_x_est:
            est = [est[i].unsqueeze(0) for i in est_ids]
            est = torch.cat(est, dim=0)
            est = rearrange(est, "t b w h c -> (t b) w h c")
            return traj[-1], est
        else:
            return traj[-1]
    # 在编辑过程中使用欧拉采样
    @torch.no_grad()
    def sample_euler_replacement_edit_till(
        self, model, z_orig, N, edit_till, model_kwargs=None, ode_kwargs=None
    ):
        inpainting_mask, inpainted_motion = (
            model_kwargs["y"]["inpainting_mask"],
            model_kwargs["y"]["inpainted_motion"],
        )

        dt = 1.0 / N
        traj = []  # to store the trajectory
        z = z_orig.detach().clone()
        batchsize = len(z)

        est = []
        return_x_est = ode_kwargs["return_x_est"]
        if return_x_est:
            return_x_est_num = ode_kwargs["return_x_est_num"]
            est_ids = [int(i * N / return_x_est_num) for i in range(return_x_est_num)]

        traj.append(z.detach().clone())
        for i in range(0, N, 1):
            t = torch.ones((batchsize), device=z_orig.device) * i / N

            _inpainted_motion = (z_orig * (N - i) + inpainted_motion * i) / N
            if i * 1.0 / N <= edit_till:
                z = (z * ~inpainting_mask) + (_inpainted_motion * inpainting_mask)

            pred = model(z, t, **model_kwargs)

            _est_now = z + (1 - i * 1.0 / N) * pred
            est.append(_est_now.detach().clone())

            z = z.detach().clone() + pred * dt
            traj.append(z.detach().clone())

        if return_x_est:
            est = [est[i].unsqueeze(0) for i in est_ids]
            est = torch.cat(est, dim=0)
            est = rearrange(est, "t b w h c -> (t b) w h c")
            return traj[-1], est
        else:
            return traj[-1]
    # 计算曲线的弯曲度
    @torch.no_grad()
    def cal_curveness(self, model, z_orig, N, model_kwargs):
        print(f"cal_curveness, N={N}")
        dt = 1.0 / N
        traj = []  # to store the trajectory
        preds = []
        z = z_orig.detach().clone()
        bs = len(z)

        func = lambda t, x: model(x, t, **model_kwargs)
        target = (
            odeint(
                func,
                z,
                # 0.0,
                torch.tensor([0.0, 1.0], device=z_orig.device, dtype=z_orig.dtype),
                # phi=self.parameters(),
                rtol=1e-5,
                atol=1e-5,
                method="dopri5",
                adjoint_params=(),
                # **ode_kwargs
                # options=dict(step_size=1/100),
            )[-1]
            .detach()
            .clone()
        )
        # pre-compute the target, as it's too memory-consuming to save the intermediate results

        traj.append(z.detach().clone())
        for i in tqdm(range(0, N, 1), desc="cal_curveness", total=N):
            t = torch.ones(bs, device=z_orig.device) * i / N
            pred = model(z, t, **model_kwargs)
            pred = pred.detach().clone()
            preds.append((pred - target).pow(2).mean().item())
            z = z.detach().clone() + pred * dt
            traj.append(z.detach().clone())

        result = sum(preds) / len(preds)
        print("curveness: ", result)
        return result

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        ode_kwargs=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device="cuda",
        progress=False,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
        cond_fn_with_grad=False,
        dump_steps=None,
        const_noise=False,
        sample_steps=None,  # backward compatibility, never use it
    ):
        if noise is None:
            # print("noise is None, use randn instead")
            noise = torch.randn(*shape, device=device)
        func = lambda t, x: model(x, t, **model_kwargs)
        if ode_kwargs["method"] in ["euler", "dopri5"]:
            assert not ("return_x_est" in ode_kwargs and ode_kwargs["return_x_est"])
            if ode_kwargs["method"] == "euler":
                ode_kwargs = dict(
                    rtol=ode_kwargs["rtol"],
                    atol=ode_kwargs["atol"],
                    method="euler",
                    options=dict(step_size=ode_kwargs["step_size"]),
                )
            elif ode_kwargs["method"] == "dopri5":
                ode_kwargs = dict(
                    rtol=ode_kwargs["rtol"],
                    atol=ode_kwargs["atol"],
                    method="dopri5",
                )
            data = odeint(
                func,
                noise,
                # 0.0,
                torch.tensor([0.0, 1.0], device=device, dtype=noise.dtype),
                # phi=self.parameters(),
                # method="euler", # "dopri5",
                # rtol=1e-5,
                # atol=1e-5,
                adjoint_params=(),
                **ode_kwargs
                # options=dict(step_size=1/100),
            )
            data = data[-1]
        elif ode_kwargs["method"] == "euler_replacement_edit_till":
            data = self.sample_euler_replacement_edit_till(
                model,
                z_orig=noise,
                N=int(1 / ode_kwargs["step_size"]),
                edit_till=ode_kwargs["edit_till"],
                model_kwargs=model_kwargs,
                ode_kwargs=ode_kwargs,
            )

        elif ode_kwargs["method"] == "odenoise_euler_replacement":
            inpainting_mask, inpainted_motion = (
                model_kwargs["y"]["inpainting_mask"],
                model_kwargs["y"]["inpainted_motion"],
            )
            partial_data = (0 * ~inpainting_mask) + (inpainted_motion * inpainting_mask)
            _ode_kwargs = dict(
                rtol=ode_kwargs["rtol"],
                atol=ode_kwargs["atol"],
                method="dopri5",
            )
            noise = odeint(
                func,
                partial_data,
                # 0.0,
                torch.tensor([1.0, 0.0], device=device, dtype=noise.dtype),
                # phi=self.parameters(),
                # method="euler", # "dopri5",
                # rtol=1e-5,
                # atol=1e-5,
                adjoint_params=(),
                **_ode_kwargs
                # options=dict(step_size=1/100),
            )[-1]

            data = self.sample_euler_replacement_edit_till(
                model,
                z_orig=noise,
                N=int(1 / ode_kwargs["step_size"]),
                edit_till=ode_kwargs["edit_till"],
                model_kwargs=model_kwargs,
                ode_kwargs=ode_kwargs,
            )
        elif ode_kwargs["method"] == "variation_euler_replacement":
            inpainting_mask, inpainted_motion = (
                model_kwargs["y"]["inpainting_mask"],
                model_kwargs["y"]["inpainted_motion"],
            )
            partial_data = inpainted_motion  # (0 * ~inpainting_mask) + (inpainted_motion * inpainting_mask)
            _ode_kwargs = dict(
                rtol=ode_kwargs["rtol"],
                atol=ode_kwargs["atol"],
                method="dopri5",
            )
            noise = odeint(
                func,
                partial_data,
                # 0.0,
                torch.tensor([1.0, 0.0], device=device, dtype=noise.dtype),
                # phi=self.parameters(),
                # method="euler", # "dopri5",
                # rtol=1e-5,
                # atol=1e-5,
                adjoint_params=(),
                **_ode_kwargs
                # options=dict(step_size=1/100),
            )[-1]
            _noise_masked = (torch.randn_like(noise) * ~inpainting_mask) + (
                noise * inpainting_mask
            )

            data = self.sample_euler_replacement(
                model,
                z_orig=_noise_masked,
                N=int(1 / ode_kwargs["step_size"]),
                model_kwargs=model_kwargs,
            )
        elif ode_kwargs["method"] == "euler_raw":
            data = self.sample_euler_raw(
                model,
                z_orig=noise,
                N=int(1 / ode_kwargs["step_size"]),
                ode_kwargs=ode_kwargs,
                model_kwargs=model_kwargs,
            )
        else:
            raise NotImplementedError

        is_return_est = isinstance(data, tuple)
        if is_return_est:
            data, x0_est = data
            assert model.training is not True, "x0_est is only for inference"

        data_range_dict = dict()
        # 0th-joint as an obversation
        data_range_dict["data_range_j0/gen_mean"] = data[:, 0].mean()
        data_range_dict["data_range_j0/gen_std"] = data[:, 0].std()
        data_range_dict["data_range_j0/gen_min"] = data[:, 0].min()
        data_range_dict["data_range_j0/gen_max"] = data[:, 0].max()
        if is_return_est:
            return data, x0_est
        else:
            return data

    def training_losses(
        self,
        model,
        x_start,
        t,
        model_kwargs=None,
        noise=None,
        dataset=None,
        sigma_min=1e-4,
    ):
        # mask = model_kwargs["y"]["mask"]
        # get_xyz = lambda sample: model.module.rot2xyz(
        #     sample,
        #     mask=None,
        #     pose_rep=model.module.pose_rep,
        #     translation=model.module.translation,
        #     glob=model.module.glob,
        #     # jointstype='vertices',  # 3.4 iter/sec # USED ALSO IN MotionCLIP
        #     jointstype="smpl",  # 3.4 iter/sec
        #     vertstrans=False,
        # )

        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = torch.randn_like(x_start)
        assert t is None
        t = torch.rand(len(x_start), device=x_start.device, dtype=x_start.dtype)
        t_1d = t[:,]  # [B, 1, 1, 1]
        t = t[:, None]  # [B, 1, 1, 1]
        x_t = t * x_start + (1 - (1 - sigma_min) * t) * noise
        target = x_start - (1 - sigma_min) * noise

        terms = {}
        model_output = model(x_t, t_1d, **model_kwargs)
        terms["rot_mse"] = self.l2_loss(
            target, model_output
        )  # mean_flat(rot_mse)

        terms["loss"] = (
            terms["rot_mse"]
        )
        return terms
