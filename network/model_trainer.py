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
        _dataset = ImageDataset(resolution=self.image_size,
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
        bdr=batch["bdr"]

        loss = self.model.training_loss(
            img, image_features, text_feature, projection_matrix, kernel_size=kernel_size, cond=cond,bdr=bdr).mean()

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