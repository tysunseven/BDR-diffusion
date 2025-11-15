import fire
import os
from network.model_trainer import DiffusionModel
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import seed_everything
from pytorch_lightning.plugins import DDPPlugin
from utils.utils import exists
from pytorch_lightning import loggers as pl_loggers
from utils.utils import ensure_directory, run, get_tensorboard_dir, find_best_epoch


def train_from_folder(
    img_folder: str = "/home/D/dataset/",
    data_class: str = "chair",
    results_folder: str = './results',
    name: str = "model",
    image_size: int = 64,
    base_channels: int = 32,
    optimizier: str = "adam",
    attention_resolutions: str = "4, 8",
    lr: float = 2e-4,
    batch_size: int = 4,
    with_attention: bool = True,
    num_heads: int = 4,
    dropout: float = 0.1,
    noise_schedule: str = "linear",
    kernel_size: float = 2.0,
    ema_rate: float = 0.999,
    save_last: bool = True,
    verbose: bool = False,
    training_epoch: int = 200,
    in_azure: bool = False,
    continue_training: bool = False,
    debug: bool = False,
    seed: int = 777,
    save_every_epoch: int = 20,
    gradient_clip_val: float = 1.,
    feature_drop_out: float = 0.1,
    data_augmentation: bool = False,
    view_information_ratio: float = 2.0,
    vit_global: bool = False,
    vit_local: bool = True,
    split_dataset: bool = False,
    elevation_zero: bool = False,
    detail_view: bool = False,
    run_timestamp: str = None  # <--- 在这里添加新参数
):
    if not in_azure:
        debug = True
    else:
        debug = False

    data_classes = []
    data_classes.extend(["debug","microstructure", "all"])
    assert data_class in data_classes

    

    # 在命名中增加时间戳
    # 如果没有从 shell 传入时间戳，则自动生成一个
    # (但这在 DDP 中仍会产生两个文件夹，所以强烈建议从 train.sh 传入)
    if run_timestamp is None:
        from datetime import datetime
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print("Warning: run_timestamp not provided, generating one. This may cause issues in DDP.")

    results_folder = results_folder + "/" + name + "_" + str(run_timestamp)

    rank = os.environ.get("LOCAL_RANK", "0")
    # 打印rank
    # print(f"Local Rank: {rank}")

    # 只有主进程 (rank 0) 才执行文件系统操作
    if rank == "0":
        ensure_directory(results_folder)

    model_args = dict(
        results_folder=results_folder,
        img_folder=img_folder,
        data_class=data_class,
        batch_size=batch_size,
        lr=lr,
        image_size=image_size,
        noise_schedule=noise_schedule,
        base_channels=base_channels,
        optimizier=optimizier,
        attention_resolutions=attention_resolutions,
        with_attention=with_attention,
        num_heads=num_heads,
        dropout=dropout,
        ema_rate=ema_rate,
        verbose=verbose,
        save_every_epoch=save_every_epoch,
        kernel_size=kernel_size,
        training_epoch=training_epoch,
        gradient_clip_val=gradient_clip_val,
        debug=debug,
        image_feature_drop_out=feature_drop_out,
        view_information_ratio=view_information_ratio,
        data_augmentation=data_augmentation,
        vit_global=vit_global,
        vit_local=vit_local,
        split_dataset=split_dataset,
        elevation_zero=elevation_zero,
        detail_view=detail_view
    )
    seed_everything(seed)

    model = DiffusionModel(**model_args)

    if in_azure:
        try:
            log_dir = get_tensorboard_dir()
        except Exception as e:
            log_dir = results_folder
    else:
        log_dir = results_folder

    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=log_dir,
        version=None,
        name='logs',
        default_hp_metric=False
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="current_epoch",
        dirpath=results_folder,
        filename="{epoch:02d}",
        save_top_k=10,
        save_last=save_last,
        every_n_epochs=save_every_epoch,
        mode="max",
    )

    last_epoch = find_best_epoch(results_folder)
    if os.path.exists(os.path.join(results_folder, "last.ckpt")):
        last_ckpt = "last.ckpt"
    else:
        if exists(last_epoch):
            last_ckpt = f"epoch={last_epoch:02d}.ckpt"
        else:
            last_ckpt = "last.ckpt"

    find_unused_parameters = False
    if in_azure:
        trainer = Trainer(devices=-1,
                          accelerator="gpu",
                          strategy=DDPPlugin(
                              find_unused_parameters=find_unused_parameters),
                          logger=tb_logger,
                          max_epochs=training_epoch,
                          log_every_n_steps=10,
                          callbacks=[checkpoint_callback])
    else:
        trainer = Trainer(devices=-1,
                          accelerator="gpu",
                          strategy=DDPPlugin(
                              find_unused_parameters=find_unused_parameters),
                          logger=tb_logger,
                          max_epochs=training_epoch,
                          log_every_n_steps=1,
                          callbacks=[checkpoint_callback])

    if continue_training and os.path.exists(os.path.join(results_folder, last_ckpt)):
        trainer.fit(model, ckpt_path=os.path.join(results_folder, last_ckpt))
    else:
        trainer.fit(model)


if __name__ == '__main__':
    fire.Fire(train_from_folder)
