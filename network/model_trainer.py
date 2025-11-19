import copy
from utils.utils import set_requires_grad
from torch.utils.data import DataLoader
from network.model_utils import EMA,make_sym,noise_sym,noise_sym_like
from network.data_loader import ImageDataset, AcousticDataset
from pathlib import Path
from torch.optim import AdamW,Adam
from utils.utils import update_moving_average
from pytorch_lightning import LightningModule
from network.model import myDiffusion, AcousticDiffusion
import torch.nn as nn
import os
import random


class DiffusionModel(LightningModule):
    def __init__(
        self,
        img_folder: str = "",
        data_class: str = "chair",
        results_folder: str = './results',
        image_size: int = 32,
        base_channels: int = 32,
        lr: float = 2e-4,
        batch_size: int = 8,
        attention_resolutions: str = "16,8",
        optimizier: str = "adam",
        with_attention: bool = False,
        num_heads: int = 4,
        dropout: float = 0.0,
        ema_rate: float = 0.999,
        verbose: bool = False,
        save_every_epoch: int = 1,
        training_epoch: int = 100,
        gradient_clip_val: float = 1.0,
        noise_schedule: str = "linear",
        debug: bool = False,
        image_feature_drop_out: float = 0.1,
        view_information_ratio: float = 0.5,
        data_augmentation: bool = False,
        kernel_size: float = 2.0,
        vit_global: bool = False,
        vit_local: bool = True,
        split_dataset: bool = False,
        elevation_zero: bool = False,
        detail_view: bool = False,
    ):

        super().__init__()
        self.save_hyperparameters()

        self.automatic_optimization = False
        self.results_folder = Path(results_folder)
        self.model = myDiffusion(image_size=image_size, base_channels=base_channels,
                                        attention_resolutions=attention_resolutions,
                                        with_attention=with_attention,
                                        kernel_size=kernel_size,
                                        dropout=dropout,
                                        num_heads=num_heads,
                                        noise_schedule=noise_schedule,
                                        vit_global=vit_global,
                                        vit_local=vit_local,
                                        verbose=verbose)

        self.view_information_ratio = view_information_ratio
        self.kernel_size = kernel_size
        self.batch_size = batch_size
        self.lr = lr
        self.image_size = image_size
        self.img_folder = img_folder
        self.data_class = data_class
        self.data_augmentation = data_augmentation
        self.with_attention = with_attention
        self.save_every_epoch = save_every_epoch
        self.traning_epoch = training_epoch
        self.gradient_clip_val = gradient_clip_val
        # 创建了一个EMA类的实例，并将其赋值给self.ema_updater属性
        self.ema_updater = EMA(ema_rate)
        # 创建主模型 self.model 的一个“影子副本”
        self.ema_model = copy.deepcopy(self.model)
        self.image_feature_drop_out = image_feature_drop_out

        self.vit_global = vit_global
        self.vit_local = vit_local
        self.split_dataset = split_dataset
        self.elevation_zero = elevation_zero
        self.detail_view = detail_view
        self.optimizier = optimizier
        # 在上面已经有了self.ema_model = copy.deepcopy(self.model) 这一步骤
        # 所以 self.reset_parameters() 在这里是冗余的，但它不会引起任何错误
        self.reset_parameters()
        # 设置 ema_model 不需要计算梯度，因为它只是一个平均值的“容器”，不参与反向传播
        set_requires_grad(self.ema_model, False)
        if debug:
            self.num_workers = 1
        else:
            self.num_workers = os.cpu_count()
    
    # 将主模型（self.model）的权重（参数）复制到 EMA “影子”模型（self.ema_model）中
    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def update_EMA(self):
        # 本质上就是对模型中的每个参数使用一遍update_average
        # 上面初始化的self.ema_updater对象被传递进来
        update_moving_average(self.ema_model, self.model, self.ema_updater)

    def configure_optimizers(self):
        # self.optimizier 是一个字符串 (string) 变量，它指定了要使用的优化器类型
        # optimizer 被用来存储 AdamW 或 Adam 类的实例，才是真正的优化器对象
        # optimizier 这个拼写看起来是一个拼写错误，正确的拼写应该是 optimizer
        # configure_optimizers 方法和 optimizers 方法都是LightningModule 的标准方法
        # configure_optimizers 是一个钩子，作为开发者需要覆盖 (override) 和实现 (implement) 它
        # 调用它的人是 PyTorch Lightning 框架，在训练开始时，框架会自动调用这个方法一次
        # 它的工作是创建、定义并返回模型将要使用的优化器（或优化器列表）
        # optimizers 方法则是 LightningModule 的一个辅助方法
        # 调用者是我也就是开发者，通常在 training_step 内部被调用
        # 特别是automatic_optimization 设置为 False 时
        # 它的工作是从 Lightning 框架那里取回你之前由 configure_optimizers 定义的优化器实例
        if self.optimizier == "adamw":
            optimizer = AdamW(self.model.parameters(), lr=self.lr)
        elif self.optimizier == "adam":
            optimizer = Adam(self.model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError
        return [optimizer]

    def train_dataloader(self):
        # 钩子方法，当 trainer.fit(model) 开始时，框架会自动调用这个方法，以获取用于训练的数据加载器（DataLoader）
        _dataset = ImageDataset(resolution=self.image_size,
                                data_folder=self.img_folder,)
        dataloader = DataLoader(_dataset,
                                # num_workers=self.num_workers,
                                num_workers=4,
                                batch_size=self.batch_size, shuffle=True, pin_memory=True, drop_last=False)
        self.iterations = len(dataloader)
        return dataloader

    def training_step(self, batch, batch_idx):
        # 钩子方法，在训练过程中，框架会为 train_dataloader 中的每一个批次 (batch) 的数据都调用一次这个方法
        # 来自于 data_loader.py 中 ImageDataset 的 __getitem__ 方法的返回值
        # batch 是一个字典，它的键包括 "img"、"cond" 和 "bdr"，于是下面的命令获得了对应的值
        img=batch["img"]
        cond=batch["cond"]
        bdr=batch["bdr"]

        # 这四个参数被传递给 model.training_loss 方法
        # 注意 model 是 myDiffusion 类的一个实例，而不是 DiffusionModel 类的实例
        image_features = None
        text_feature = None
        projection_matrix = None
        kernel_size = None

        # self.model.training_loss(...) 方法返回的是一个 PyTorch 张量 (Tensor)
        # PyTorch 张量 (Tensor)的核心特性是会携带“历史记录”（计算图），用于自动求导
        # .mean() 是 PyTorch 张量的一个方法，用于计算张量中所有元素的平均值
        # .mean() 之后的结果是一个标量，但仍然是一个 PyTorch 张量，仍然携带计算图
        loss = self.model.training_loss(
            img, image_features, text_feature, projection_matrix, kernel_size=kernel_size, cond=cond,bdr=bdr).mean()

        # 这里的原代码是 self.log("loss", loss.clone().detach().item(), prog_bar=True)
        # 这里做了改动是为了在tensorboard中增加关于epoch的loss曲线
        # 关于 self.log 方法的更多细节之后还需要花时间去研究
        self.log("loss", loss.detach(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # optimizers()和configure_optimizers() 方法的关系见上面的注释
        opt = self.optimizers()
        # PyTorch 在计算梯度时，默认会累加（accumulate）梯度
        # 假设你希望用一个很大的 batch_size（例如 1024）来训练模型
        # 因为大的批次通常能让模型训练更稳定
        # 但是，你的 GPU 显存非常有限，一次最多只能处理 batch_size 为 8 的数据
        # 可以把那个大小为 1024 的“大批次”拆分成 128 个“微批次”（micro-batch），每个“微批次”的大小为 8
        # 在开始大批次之前，先调用一次 opt.zero_grad()（确保梯度是干净的）
        # 因为 PyTorch 默认是累加梯度，所以在循环结束后，模型所有参数 (self.model.parameters()) 上存储的梯度
        # 已经是全部 128 个“微批次”的梯度之和——这等效于你用 batch_size 为 1024 计算出来的梯度
        # 优化器会使用这个累加了 128 次的“大梯度”来更新一次模型权重
        # 这也就是为什么这里我们需要调用 opt.zero_grad() 来清零梯度
        opt.zero_grad()
        # loss.backward()：是 PyTorch 原生的、基础的反向传播指令
        # self.manual_backward(loss)：是 PyTorch Lightning 提供的“智能包装器” (wrapper)
        # 使用 self.manual_backward 是为了确保在使用 Lightning 的自动混合精度 (AMP) 或分布式训练 (DDP) 时
        # 反向传播过程能够正确地处理这些复杂的场景，否则需要自己手动实现 GradScaler 和梯度同步
        # 这里关闭自动优化是因为要插入梯度裁剪 (gradient clipping) 逻辑
        self.manual_backward(loss)
        # 检查所有梯度，计算它们的“总范数”（Norm，可以理解为梯度的总体大小）
        # 如果“总范数”超过了 self.gradient_clip_val（一个预设的阈值）
        # 那么就按比例缩小所有梯度，确保“总范数”等于 self.gradient_clip_val
        # 防止模型因为某一次 batch 算出了过大的梯度而导致训练（权重更新）“飞出”稳定区域
        # gradient_clip_val 是定义 DiffusionModel 类实例时传入的一个参数，默认值是 1.0
        # nn.utils.clip_grad_norm_ 是一个 PyTorch 框架中已经内置好的、非常标准的辅助函数
        # 它主要接收两个参数，希望裁剪的模型参数和梯度的“范数”上限
        nn.utils.clip_grad_norm_(
            self.model.parameters(), self.gradient_clip_val)
        opt.step()
        # 在每个训练步骤中，在优化器更新完主模型（opt.step()）之后，self.update_EMA() 会被调用
        # 这确保了 ema_model 总是基于 model 的最新权重来计算平滑平均值
        # EMA 的最终目的是在generate.py 中被使用
        # 当加载模型用于生成时，generate.py 脚本会检查 ema 标志（默认为 True）
        # 如果 ema 为 True，那么用于生成图像的 generator 就会被设置为 discrete_diffusion.ema_model
        # （即那个平滑的“影子副本”），而不是 discrete_diffusion.model（主训练模型）
        self.update_EMA()

    def on_train_epoch_end(self):
        # 钩子方法，框架会在每一个训练周期 (epoch) 结束时调用一次这个方法
        self.log("current_epoch", self.current_epoch, logger=False)
        # 调用父类的 on_train_epoch_end 方法，确保任何在父类中定义的逻辑也会被执行
        # 确保了从父类返回的值能被原封不动地传递回给最初的调用者——“训练器”
        return super().on_train_epoch_end()

class AcousticDiffusionModel(LightningModule):
    """
    用于声学超材料 (Acoustic Metamaterial) 的 Lightning 模块。
    - 移除: BDR (边界条件) 逻辑。
    - 移除: 对称性约束 (Symmetry) 逻辑。
    - 更改: 使用 AcousticDataset 和 AcousticDiffusion。
    """
    def __init__(
        self,
        img_folder: str = "",
        data_class: str = "chair",
        results_folder: str = './results',
        image_size: int = 32,
        base_channels: int = 32,
        lr: float = 2e-4,
        batch_size: int = 8,
        attention_resolutions: str = "16,8",
        optimizier: str = "adam",
        with_attention: bool = False,
        num_heads: int = 4,
        dropout: float = 0.0,
        ema_rate: float = 0.999,
        verbose: bool = False,
        save_every_epoch: int = 1,
        training_epoch: int = 100,
        gradient_clip_val: float = 1.0,
        noise_schedule: str = "linear",
        debug: bool = False,
        image_feature_drop_out: float = 0.1,
        view_information_ratio: float = 0.5,
        data_augmentation: bool = False,
        kernel_size: float = 2.0,
        vit_global: bool = False,
        vit_local: bool = True,
        split_dataset: bool = False,
        elevation_zero: bool = False,
        detail_view: bool = False,
    ):

        super().__init__()
        self.save_hyperparameters()

        self.automatic_optimization = False
        self.results_folder = Path(results_folder)
        
        # 更改: 使用 AcousticDiffusion
        self.model = AcousticDiffusion(image_size=image_size, base_channels=base_channels,
                                        attention_resolutions=attention_resolutions,
                                        with_attention=with_attention,
                                        kernel_size=kernel_size,
                                        dropout=dropout,
                                        num_heads=num_heads,
                                        noise_schedule=noise_schedule,
                                        vit_global=vit_global,
                                        vit_local=vit_local,
                                        verbose=verbose)

        self.view_information_ratio = view_information_ratio
        self.kernel_size = kernel_size
        self.batch_size = batch_size
        self.lr = lr
        self.image_size = image_size
        self.img_folder = img_folder
        self.data_class = data_class
        self.data_augmentation = data_augmentation
        self.with_attention = with_attention
        self.save_every_epoch = save_every_epoch
        self.traning_epoch = training_epoch
        self.gradient_clip_val = gradient_clip_val
        self.ema_updater = EMA(ema_rate)
        self.ema_model = copy.deepcopy(self.model)
        self.image_feature_drop_out = image_feature_drop_out

        self.vit_global = vit_global
        self.vit_local = vit_local
        self.split_dataset = split_dataset
        self.elevation_zero = elevation_zero
        self.detail_view = detail_view
        self.optimizier = optimizier
        self.reset_parameters()
        set_requires_grad(self.ema_model, False)
        if debug:
            self.num_workers = 1
        else:
            self.num_workers = os.cpu_count()

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def update_EMA(self):
        update_moving_average(self.ema_model, self.model, self.ema_updater)

    def configure_optimizers(self):
        if self.optimizier == "adamw":
            optimizer = AdamW(self.model.parameters(), lr=self.lr)
        elif self.optimizier == "adam":
            optimizer = Adam(self.model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError
        return [optimizer]

    def train_dataloader(self):
        # 更改: 使用 AcousticDataset
        _dataset = AcousticDataset(resolution=self.image_size,
                                data_folder=self.img_folder,)
        dataloader = DataLoader(_dataset,
                                # num_workers=self.num_workers,
                                num_workers=4,
                                batch_size=self.batch_size, shuffle=True, pin_memory=True, drop_last=False)
        self.iterations = len(dataloader)
        return dataloader

    def training_step(self, batch, batch_idx):
        image_features = None
        projection_matrix = None
        kernel_size = None
        text_feature = None
        
        img=batch["img"]
        cond=batch["cond"]
        # 移除: bdr=batch["bdr"]

        # 更改: 移除 'bdr' 参数
        loss = self.model.training_loss(
            img, image_features, text_feature, projection_matrix, kernel_size=kernel_size, cond=cond).mean()

        self.log("loss", loss.detach(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        nn.utils.clip_grad_norm_(
            self.model.parameters(), self.gradient_clip_val)
        opt.step()

        self.update_EMA()

    def on_train_epoch_end(self):
        self.log("current_epoch", self.current_epoch, logger=False)
        return super().on_train_epoch_end()