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
        # img_features, text_feature, projection_matrix, kernel_size 这些参数虽然被定义，但传进来的值都是 None
        # 所以有用的也就只有 img, cond, bdr 这三个参数了，也就是来自于 data_loader.py 中 ImageDataset 的 __getitem__ 方法的返回值
        # 从输入的图像张量 img 中获取批次大小（batch size），即这个批次里有多少张图片
        batch = img.shape[0]

        # 随机将当前批次中 1/8 样本的条件 cond 强制设置为 -1。-1 在这里充当一个“空”或“无条件”的标记
        # 让同一个模型既学习“有条件”的去噪，也学习“无条件”的去噪
        # 在生成时，通过放大这两者之间的差异来强化“有条件”的特征
        cond[-int(batch/8):,:]=-1

        # torch.zeros((batch,), device=self.device) 创建了一个一维张量，长度为 batch，所有元素初始化为 0，并且这个张量被分配到与模型相同的设备上（CPU 或 GPU）
        # .float().uniform_(0, 1) 将这个张量的数据类型转换为浮点数，并用均匀分布在 [0, 1) 范围内的随机数填充它
        times = torch.zeros(
            (batch,), device=self.device).float().uniform_(0, 1)
        
        # noise = torch.randn_like(img)
        # 这个函数的定义在 model_utils.py 中
        # 生成一个形状与 img 相同、但具有特定对称性的噪声
        # 在我的问题里使用普通的噪声就好
        noise = noise_sym_like(img)

        # 根据随机生成的时间 t, 计算对应的 log SNR（信噪比的对数）
        noise_level = self.log_snr(times)

        # 将 noise_level 的形状调整为与 img 兼容
        padded_noise_level = right_pad_dims_to(img, noise_level)

        # 将“对数信噪比”转换为“信号缩放因子” $alpha$ 和“噪声缩放因子” $sigma$
        alpha, sigma = log_snr_to_alpha_sigma(padded_noise_level)

        # 根据扩散模型的前向过程，向原始图像 img 添加噪声，生成 noised_img
        noised_img = alpha * img + sigma * noise

        # 初始化“自条件” (self-conditioning) 变量为空
        self_cond = None

        # 以 50% 的概率启用自条件
        # img_features, text_feature, projection_matrix, kernel_size 这些参数被传入，但依然是None
        if random() < 0.5:
            with torch.no_grad():
                self_cond = self.denoise_fn(
                    noised_img, noise_level, img_features, text_feature, projection_matrix, kernel_size=kernel_size, cond=cond,bdr=bdr).detach_()
                self_cond=make_sym(self_cond,device=self.device)
        pred = self.denoise_fn(noised_img, noise_level,
                               img_features, text_feature, projection_matrix, self_cond, kernel_size=kernel_size, cond=cond,bdr=bdr)

        return F.mse_loss(pred, img)

    # 这是一个生成部分的采样函数
    @torch.no_grad()
    def sample_conditional_bdr_json(self, batch_size=16,
                             steps=50, truncated_index: float = 0.0, verbose: bool = True, C=None, mybdr=None):
        image_size = self.image_size
        shape = (batch_size, 1, image_size, image_size)
        
        # from utils.condition_data import white_image_feature, an_object_feature
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


    # 用于调试和可视化的函数。它执行标准的去噪生成过程，但额外承担了“录像”的任务，将每一步的变化都保存到了硬盘上的 .npz 文件中
    @torch.no_grad()
    def sample_process(self, batch_size=16,
                             steps=50, truncated_index: float = 0.0, verbose: bool = True, C=None, mybdr=None):
        image_size = self.image_size
        shape = (batch_size, 1, image_size, image_size)
        
        # from utils.condition_data import white_image_feature, an_object_feature
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
        # 在我的代码里把值设为-1应该不太合适
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