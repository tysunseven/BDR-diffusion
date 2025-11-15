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

    # 显式地将 fire 传入的(字符串)"false"或(布尔)False 转换为真正的布尔值
    # 我们只关心它是否是字面意义上的 True 或 "true"
    continue_training = (str(continue_training).lower() == 'true')
    

    # 验证 run_timestamp 是否被正确提供
    if run_timestamp is None:
        if continue_training is True:
            # 这是一个错误：继续训练时必须提供时间戳
            raise ValueError("错误: --continue_training=True, 但没有提供 --run_timestamp。")
        else:
            # 这是一个新训练，但 train.sh 脚本没有生成时间戳 (兜底情况)
            from datetime import datetime
            run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            print(f"警告: 正在开始新训练，但未从 train.sh 收到 --run_timestamp。")
            print(f"警告: 自动生成了一个时间戳: {run_timestamp}。这在DDP中可能不安全。")


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

    rank = os.environ.get('LOCAL_RANK', '0')
    ckpt_to_load_path = None  # 最终决定要加载的检查点路径

    if continue_training:
        print(f"--- [Rank {rank} 检查点调试] (模式: 继续训练) ---")
        
        # --- 优先级 1: 检查 'last.ckpt' ---
        potential_last_ckpt = os.path.join(results_folder, "last.ckpt")
        print(f"[Rank {rank} 调试] 优先级 1: 正在检查 'last.ckpt' at: {potential_last_ckpt}")
        
        if os.path.exists(potential_last_ckpt):
            ckpt_to_load_path = potential_last_ckpt
            print(f"[Rank {rank} 调试] 找到了 'last.ckpt'。将从这里恢复。")
        else:
            # --- 优先级 2: 查找 'best' epoch ---
            print(f"[Rank {rank} 调试] 'last.ckpt' 未找到。")
            print(f"[Rank {rank} 调试] 优先级 2: 正在调用 find_best_epoch...")
            
            last_epoch_num = find_best_epoch(results_folder)
            print(f"[Rank {rank} 调试] find_best_epoch 返回: {last_epoch_num}")

            if exists(last_epoch_num): 
                ckpt_filename = f"epoch={last_epoch_num:02d}.ckpt"
                potential_best_ckpt = os.path.join(results_folder, ckpt_filename)
                print(f"[Rank {rank} 调试] 正在检查 'best' epoch: {potential_best_ckpt}")

                if os.path.exists(potential_best_ckpt):
                    ckpt_to_load_path = potential_best_ckpt
                    print(f"[Rank {rank} 调试] 找到了 'best' epoch ({ckpt_filename})。将从这里恢复。")
                else:
                    print(f"[Rank {rank} 调试] 警告: find_best_epoch 返回 {last_epoch_num}, 但文件 {ckpt_filename} 不存在!")
                    print(f"[Rank {rank} 调试] (请确认 ModelCheckpoint 的 filename 已修正为 'epoch=...')")
            else:
                 print(f"[Rank {rank} 调试] find_best_epoch 未返回任何 epoch。")
        
        if ckpt_to_load_path is None:
            print(f"[Rank {rank} 调试] 警告: 'continue_training' 为 True, 但在 {results_folder} 中未找到任何可加载的检查点!")
        
        print(f"--- [Rank {rank} 结束检查点调试] ---")
    
    else:
        print(f"--- [Rank {rank} 检查点调试] (模式: 新训练, 将从头开始) ---")


    # --- [新的] 启动 trainer.fit() 的逻辑 ---
    # 只有当 'continue_training' 为 True 且我们成功找到了一个检查点时, 'ckpt_to_load_path' 才会有值
    if ckpt_to_load_path:
        print(f"[Rank {rank} 最终决定] 将调用 trainer.fit(ckpt_path='{ckpt_to_load_path}')")
        trainer.fit(model, ckpt_path=ckpt_to_load_path)
    else:
        if continue_training:
            # 如果用户想继续, 但我们没找到文件, 我们必须警告他们
            print(f"[Rank {rank} 最终决定] 警告: 本应继续训练, 但未找到检查点。将从头开始训练。")
        else:
            print(f"[Rank {rank} 最终决定] 将调用 trainer.fit(model) (从头开始训练)")
        trainer.fit(model)

if __name__ == '__main__':
    fire.Fire(train_from_folder)
