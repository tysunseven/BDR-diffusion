import torch
import torch.nn.functional as F
from tqdm import tqdm
from network.model_utils import *
from network.unet import UNetModel
from network.unet import AcousticUNetModel
from einops import rearrange, repeat
from random import random
from functools import partial
from torch import nn
from torch.special import expm1
import time as tm

from PIL import Image
import json as js
import os

TRUNCATED_TIME = 0.7

def mypost_process(res_tensor):
        res_tensor = ((res_tensor+1)*127.5).clamp(0,255).to(torch.uint8)
        res_tensor = res_tensor.permute(0,2,3,1)
        res_tensor = res_tensor.contiguous()
        return res_tensor
    
def mysave_as_npz(gathered_samples,path):
    arr = np.array(gathered_samples)
    np.savez(path,arr)

class myDiffusion(nn.Module):
    def __init__(
            self,
            image_size: int = 64,
            base_channels: int = 128,
            attention_resolutions: str = "16,8",
            with_attention: bool = False,
            num_heads: int = 4,
            dropout: float = 0.0,
            verbose: bool = False,
            eps: float = 1e-6,
            noise_schedule: str = "linear",
            kernel_size: float = 1.0,
            vit_global: bool = False,
            vit_local: bool = True,
    ):
        super().__init__()
        self.image_size = image_size
        if image_size == 8:
            channel_mult = (1, 4, 8)
        elif image_size == 32:
            channel_mult = (1, 2, 4, 8)
        elif image_size == 64:
            channel_mult = (1, 2, 4, 8, 8)
        elif image_size == 128:
            channel_mult = (1, 2, 4, 8, 8)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
        attention_ds = []
        for res in attention_resolutions.split(","):
            attention_ds.append(image_size // int(res))
        self.eps = eps
        self.verbose = verbose
        if noise_schedule == "linear":
            self.log_snr = beta_linear_log_snr
        elif noise_schedule == "cosine":
            self.log_snr = alpha_cosine_log_snr
        else:
            raise ValueError(f'invalid noise schedule {noise_schedule}')
        self.denoise_fn = UNetModel(
            image_size=image_size,
            base_channels=base_channels,
            dim_mults=channel_mult, dropout=dropout,
            kernel_size=kernel_size,
            world_dims=2,
            num_heads=num_heads, vit_global=vit_global, vit_local=vit_local,
            attention_resolutions=tuple(attention_ds), with_attention=with_attention,
            verbose=verbose)
        self.vit_global = vit_global
        self.vit_local = vit_local

    @property
    def device(self):
        return next(self.denoise_fn.parameters()).device

    def get_sampling_timesteps(self, batch, device, steps):
        times = torch.linspace(1., 0., steps + 1, device=device)
        times = repeat(times, 't -> b t', b=batch)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim=0)
        times = times.unbind(dim=-1)
        return times

    def training_loss(self, img, img_features, text_feature, projection_matrix, kernel_size=None, cond=None,bdr=None, *args, **kwargs):
        batch = img.shape[0]

        # classifier-free guidance
        cond[-int(batch/8):,:]=-1

        times = torch.zeros(
            (batch,), device=self.device).float().uniform_(0, 1)
        #noise = torch.randn_like(img)
        noise = noise_sym_like(img)

        noise_level = self.log_snr(times)
        padded_noise_level = right_pad_dims_to(img, noise_level)
        alpha, sigma = log_snr_to_alpha_sigma(padded_noise_level)
        noised_img = alpha * img + sigma * noise
        self_cond = None
        # self condition
        if random() < 0.5:
            with torch.no_grad():
                self_cond = self.denoise_fn(
                    noised_img, noise_level, img_features, text_feature, projection_matrix, kernel_size=kernel_size, cond=cond,bdr=bdr).detach_()
                self_cond=make_sym(self_cond,device=self.device)
        pred = self.denoise_fn(noised_img, noise_level,
                               img_features, text_feature, projection_matrix, self_cond, kernel_size=kernel_size, cond=cond,bdr=bdr)

        return F.mse_loss(pred, img)

    @torch.no_grad()
    def sample_conditional_bdr_json(self, batch_size=16,
                             steps=50, truncated_index: float = 0.0, verbose: bool = True, C=None, mybdr=None):
        image_size = self.image_size
        shape = (batch_size, 1, image_size, image_size)
        
        from utils.condition_data import white_image_feature, an_object_feature
        batch, device = shape[0], self.device

        time_pairs = self.get_sampling_timesteps(
            batch, device=device, steps=steps)

        #img = torch.randn(shape, device=device)
        img=noise_sym(shape,device=device)
        x_start = None

        mybdr=torch.tensor(mybdr,device=device,dtype=torch.float32)


        mycond=torch.tensor(C,device=device,dtype=torch.float32)
        #mycond=torch.tensor(C,device=device,dtype=torch.float32).unsqueeze(0).repeat(int(batch),1)

        # null conditions
        null_cond = -torch.ones((batch,4),device=device)

        # guidance scale
        guidance_scale=1

        if verbose:
            _iter = tqdm(time_pairs, desc='sampling loop time step')
        else:
            _iter = time_pairs
        for time, time_next in _iter:

            log_snr = self.log_snr(time)
            log_snr_next = self.log_snr(time_next)
            log_snr, log_snr_next = map(
                partial(right_pad_dims_to, img), (log_snr, log_snr_next))

            alpha, sigma = log_snr_to_alpha_sigma(log_snr)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(log_snr_next)

            noise_cond = self.log_snr(time)

            x_null = self.denoise_fn(
                img, noise_cond, None, None, None, x_start, kernel_size=None, cond=null_cond,bdr=mybdr)
            #
            x_null=make_sym(x_null,device)
            x_start = self.denoise_fn(
                img, noise_cond, None, None, None, x_start, kernel_size=None, cond=mycond,bdr=mybdr)
            #
            x_start=make_sym(x_start,device)
            x_start = (1+guidance_scale)*x_start-guidance_scale*x_null

            if time[0] < TRUNCATED_TIME:
                x_start.sign_()
                
            c = -expm1(log_snr - log_snr_next)
            mean = alpha_next * (img * (1 - c) / alpha + c * x_start)
            variance = (sigma_next ** 2) * c
            noise = torch.where(
                rearrange(time_next > truncated_index, 'b -> b 1 1 1'),
                #torch.randn_like(img),
                noise_sym_like(img),
                torch.zeros_like(img)
            )
            img = mean + torch.sqrt(variance) * noise

        return img
    


    
    @torch.no_grad()
    def sample_process(self, batch_size=16,
                             steps=50, truncated_index: float = 0.0, verbose: bool = True, C=None, mybdr=None):
        image_size = self.image_size
        shape = (batch_size, 1, image_size, image_size)
        
        from utils.condition_data import white_image_feature, an_object_feature
        batch, device = shape[0], self.device

        time_pairs = self.get_sampling_timesteps(
            batch, device=device, steps=steps)

        #img = torch.randn(shape, device=device)
        img=noise_sym(shape,device=device)
        x_start = None

        mybdr=torch.tensor(mybdr,device=device,dtype=torch.float32)


        mycond=torch.tensor(C,device=device,dtype=torch.float32)
        #mycond=torch.tensor(C,device=device,dtype=torch.float32).unsqueeze(0).repeat(int(batch),1)

        # null conditions
        null_cond = -torch.ones((batch,4),device=device)

        # guidance scale
        guidance_scale=1

        if verbose:
            _iter = tqdm(time_pairs, desc='sampling loop time step')
        else:
            _iter = time_pairs
        gathered_samples=[]
        gathered_samples2=[]
        for time, time_next in _iter:

            log_snr = self.log_snr(time)
            log_snr_next = self.log_snr(time_next)
            log_snr, log_snr_next = map(
                partial(right_pad_dims_to, img), (log_snr, log_snr_next))

            alpha, sigma = log_snr_to_alpha_sigma(log_snr)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(log_snr_next)

            noise_cond = self.log_snr(time)

            x_null = self.denoise_fn(
                img, noise_cond, None, None, None, x_start, kernel_size=None, cond=null_cond,bdr=mybdr)
            #
            x_null=make_sym(x_null,device)
            x_start = self.denoise_fn(
                img, noise_cond, None, None, None, x_start, kernel_size=None, cond=mycond,bdr=mybdr)
            #
            x_start=make_sym(x_start,device)
            x_start = (1+guidance_scale)*x_start-guidance_scale*x_null

            if time[0] < TRUNCATED_TIME:
                x_start.sign_()
                
            c = -expm1(log_snr - log_snr_next)
            mean = alpha_next * (img * (1 - c) / alpha + c * x_start)
            variance = (sigma_next ** 2) * c
            noise = torch.where(
                rearrange(time_next > truncated_index, 'b -> b 1 1 1'),
                #torch.randn_like(img),
                noise_sym_like(img),
                torch.zeros_like(img)
            )
            gathered_samples.extend(mypost_process(mean).cpu().numpy())
            gathered_samples2.extend(mypost_process(x_start).cpu().numpy())
            img = mean + torch.sqrt(variance) * noise
        mysave_as_npz(gathered_samples,"/home/fjx/fjx/generate_process2/output2.npz")
        mysave_as_npz(gathered_samples2,"/home/fjx/fjx/generate_process2/output22.npz")
        return img

class AcousticDiffusion(nn.Module):
    """
    用于声学超材料 (Acoustic Metamaterial) 的 Diffusion 包装器。
    - 移除: BDR (边界条件) 逻辑。
    - 移除: 对称性约束 (Symmetry) 逻辑 (不再使用 make_sym, noise_sym_like)。
    - 更改: 条件(cond) 适配为 2-DOF 透射系数。
    """
    def __init__(
            self,
            image_size: int = 64,
            base_channels: int = 128,
            attention_resolutions: str = "16,8",
            with_attention: bool = False,
            num_heads: int = 4,
            dropout: float = 0.0,
            verbose: bool = False,
            eps: float = 1e-6,
            noise_schedule: str = "linear",
            kernel_size: float = 1.0,
            vit_global: bool = False,
            vit_local: bool = True,
    ):
        super().__init__()
        self.image_size = image_size
        if image_size == 8:
            channel_mult = (1, 4, 8)
        elif image_size == 32:
            channel_mult = (1, 2, 4, 8)
        elif image_size == 64:
            channel_mult = (1, 2, 4, 8, 8)
        elif image_size == 128:
            channel_mult = (1, 2, 4, 8, 8)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
        attention_ds = []
        for res in attention_resolutions.split(","):
            attention_ds.append(image_size // int(res))
        self.eps = eps
        self.verbose = verbose
        if noise_schedule == "linear":
            self.log_snr = beta_linear_log_snr
        elif noise_schedule == "cosine":
            self.log_snr = alpha_cosine_log_snr
        else:
            raise ValueError(f'invalid noise schedule {noise_schedule}')
        
        # 更改: 使用 AcousticUNetModel
        self.denoise_fn = AcousticUNetModel(
            image_size=image_size,
            base_channels=base_channels,
            dim_mults=channel_mult, dropout=dropout,
            kernel_size=kernel_size,
            world_dims=2,
            num_heads=num_heads, vit_global=vit_global, vit_local=vit_local,
            attention_resolutions=tuple(attention_ds), with_attention=with_attention,
            verbose=verbose)
        
        self.vit_global = vit_global
        self.vit_local = vit_local

    @property
    def device(self):
        return next(self.denoise_fn.parameters()).device

    def get_sampling_timesteps(self, batch, device, steps):
        times = torch.linspace(1., 0., steps + 1, device=device)
        times = repeat(times, 't -> b t', b=batch)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim=0)
        times = times.unbind(dim=-1)
        return times

    # 更改: 移除 'bdr' 参数
    def training_loss(self, img, img_features, text_feature, projection_matrix, kernel_size=None, cond=None, *args, **kwargs):
        batch = img.shape[0]

        # classifier-free guidance
        # 假设 cond 是 [B, 2], 这一操作会设置 [B_cfg, 2] 为 -1
        cond[-int(batch/8):,:]=-1

        times = torch.zeros(
            (batch,), device=self.device).float().uniform_(0, 1)
        
        # 移除对称性: 使用标准高斯噪声
        noise = torch.randn_like(img)
        # noise = noise_sym_like(img)

        noise_level = self.log_snr(times)
        padded_noise_level = right_pad_dims_to(img, noise_level)
        alpha, sigma = log_snr_to_alpha_sigma(padded_noise_level)
        noised_img = alpha * img + sigma * noise
        self_cond = None
        
        # self condition
        if random() < 0.5:
            with torch.no_grad():
                # 更改: 移除 'bdr' 参数
                self_cond = self.denoise_fn(
                    noised_img, noise_level, img_features, text_feature, projection_matrix, kernel_size=kernel_size, cond=cond).detach_()
                # 移除对称性:
                # self_cond=make_sym(self_cond,device=self.device)
                
        # 更改: 移除 'bdr' 参数
        pred = self.denoise_fn(noised_img, noise_level,
                               img_features, text_feature, projection_matrix, self_cond, kernel_size=kernel_size, cond=cond)

        return F.mse_loss(pred, img)

    @torch.no_grad()
    def sample_transmission(self, batch_size=16,
                             steps=50, truncated_index: float = 0.0, verbose: bool = True, C=None):
        """
        新的采样函数:
        - 移除 BDR
        - 移除对称性
        - 适配 2-DOF 条件 (C)
        """
        image_size = self.image_size
        shape = (batch_size, 1, image_size, image_size)
        
        # (可选导入，如果您的新模型不使用图像/文本特征，则可以移除)
        # from utils.condition_data import white_image_feature, an_object_feature
        batch, device = shape[0], self.device

        time_pairs = self.get_sampling_timesteps(
            batch, device=device, steps=steps)

        # 移除对称性: 使用标准高斯噪声
        img = torch.randn(shape, device=device)
        # img=noise_sym(shape,device=device)
        x_start = None

        # 移除: BDR 逻辑
        # mybdr=torch.tensor(mybdr,device=device,dtype=torch.float32)

        # C 的形状现在应该是 [batch, 2]
        mycond=torch.tensor(C,device=device,dtype=torch.float32)

        # null conditions
        # 更改: null_cond 适配 2-DOF
        null_cond = -torch.ones((batch, 2), device=device)
        # null_cond = -torch.ones((batch,4),device=device)

        # guidance scale
        guidance_scale=1

        if verbose:
            _iter = tqdm(time_pairs, desc='sampling loop time step')
        else:
            _iter = time_pairs
        for time, time_next in _iter:

            log_snr = self.log_snr(time)
            log_snr_next = self.log_snr(time_next)
            log_snr, log_snr_next = map(
                partial(right_pad_dims_to, img), (log_snr, log_snr_next))

            alpha, sigma = log_snr_to_alpha_sigma(log_snr)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(log_snr_next)

            noise_cond = self.log_snr(time)

            # 更改: 移除 'bdr' 参数
            x_null = self.denoise_fn(
                img, noise_cond, None, None, None, x_start, kernel_size=None, cond=null_cond)
            # 移除对称性:
            # x_null=make_sym(x_null,device)
            
            # 更改: 移除 'bdr' 参数
            x_start = self.denoise_fn(
                img, noise_cond, None, None, None, x_start, kernel_size=None, cond=mycond)
            # 移除对称性:
            # x_start=make_sym(x_start,device)
            
            x_start = (1+guidance_scale)*x_start-guidance_scale*x_null

            if time[0] < TRUNCATED_TIME:
                x_start.sign_()
                
            c = -expm1(log_snr - log_snr_next)
            mean = alpha_next * (img * (1 - c) / alpha + c * x_start)
            variance = (sigma_next ** 2) * c
            
            # 移除对称性: 使用标准高斯噪声
            noise = torch.where(
                rearrange(time_next > truncated_index, 'b -> b 1 1 1'),
                torch.randn_like(img),
                # noise_sym_like(img),
                torch.zeros_like(img)
            )
            img = mean + torch.sqrt(variance) * noise

        return img