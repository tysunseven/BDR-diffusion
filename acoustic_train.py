import fire
import os
# 更改: 导入新的 AcousticDiffusionModel
from network.model_trainer import AcousticDiffusionModel
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import seed_everything
from pytorch_lightning.plugins import DDPPlugin
from utils.utils import exists
from pytorch_lightning import loggers as pl_loggers
from utils.utils import ensure_directory, run, get_tensorboard_dir, find_best_epoch


def train_from_folder(
    img_folder: str = "/home/D/dataset/",
    # --- 新增参数 ---
    train_structures_path: str = None,
    train_properties_path: str = None,
    val_structures_path: str = None,
    val_properties_path: str = None,
    # ----------------
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
    run_timestamp: str = None
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
            print(f"警告: Geling Kai Shi Xin Xun Lian ，Dan Wei Cong train.sh Jie Shou Dao --run_timestamp。")
            print(f"警告: Zi Dong Sheng Cheng Le Yi Ge Shi Jian Chuo : {run_timestamp}。Zhe Zai DDP Zhong Ke Neng Bu An Quan 。")


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
        # --- 传入新参数 ---
        train_structures_path=train_structures_path,
        train_properties_path=train_properties_path,
        val_structures_path=val_structures_path,
        val_properties_path=val_properties_path,
        # ------------------
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

    # 更改: 实例化新的 AcousticDiffusionModel
    model = AcousticDiffusionModel(**model_args)

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
        print(f"--- [Rank {rank} Jian Cha Dian Tiao Shi ] (Mo Shi : Ji Xu Xun Lian ) ---")
        
        # --- 优先级 1: 检查 'last.ckpt' ---
        potential_last_ckpt = os.path.join(results_folder, "last.ckpt")
        print(f"[Rank {rank} Tiao Shi ] You Xian Ji 1: Zheng Zai Jian Cha 'last.ckpt' at: {potential_last_ckpt}")
        
        if os.path.exists(potential_last_ckpt):
            ckpt_to_load_path = potential_last_ckpt
            print(f"[Rank {rank} Tiao Shi ] Zhao Dao Le 'last.ckpt'。Jiang Cong Zhe Li Hui Fu 。")
        else:
            # --- 优先级 2: 查找 'best' epoch ---
            print(f"[Rank {rank} Tiao Shi ] 'last.ckpt' Wei Zhao Dao 。")
            print(f"[Rank {rank} Tiao Shi ] You Xian Ji 2: Zheng Zai Diao Yong find_best_epoch...")
            
            last_epoch_num = find_best_epoch(results_folder)
            print(f"[Rank {rank} Tiao Shi ] find_best_epoch Fan Hui : {last_epoch_num}")

            if exists(last_epoch_num): 
                ckpt_filename = f"epoch={last_epoch_num:02d}.ckpt"
                potential_best_ckpt = os.path.join(results_folder, ckpt_filename)
                print(f"[Rank {rank} Tiao Shi ] Zheng Zai Jian Cha 'best' epoch: {potential_best_ckpt}")

                if os.path.exists(potential_best_ckpt):
                    ckpt_to_load_path = potential_best_ckpt
                    print(f"[Rank {rank} Tiao Shi ] Zhao Dao Le 'best' epoch ({ckpt_filename})。Jiang Cong Zhe Li Hui Fu 。")
                else:
                    print(f"[Rank {rank} Tiao Shi ] Jing Gao : find_best_epoch Fan Hui {last_epoch_num}, Dan Wen Jian {ckpt_filename} Bu Cun Zai !")
                    print(f"[Rank {rank} Tiao Shi ] (Qing Que Ren ModelCheckpoint De filename Yi Xiu Zheng Wei 'epoch=...')")
            else:
                 print(f"[Rank {rank} Tiao Shi ] find_best_epoch Wei Fan Hui Ren He epoch。")
        
        if ckpt_to_load_path is None:
            print(f"[Rank {rank} Tiao Shi ] Jing Gao : 'continue_training' Wei True, Dan Zai {results_folder} Zhong Wei Zhao Dao Ren He Ke Jia Zai De Jian Cha Dian !")
        
        print(f"--- [Rank {rank} Jie Shu Jian Cha Dian Tiao Shi ] ---")
    
    else:
        print(f"--- [Rank {rank} Jian Cha Dian Tiao Shi ] (Mo Shi : Xin Xun Lian , Jiang Cong Tou Kai Shi ) ---")


    # --- [新的] 启动 trainer.fit() 的逻辑 ---
    # 只有当 'continue_training' 为 True 且我们成功找到了一个检查点时, 'ckpt_to_load_path' 才会有值
    if ckpt_to_load_path:
        print(f"[Rank {rank} Zui Zhong Jue Ding ] Jiang Diao Yong trainer.fit(ckpt_path='{ckpt_to_load_path}')")
        trainer.fit(model, ckpt_path=ckpt_to_load_path)
    else:
        if continue_training:
            # 如果用户想继续, 但我们没找到文件, 我们必须警告他们
            print(f"[Rank {rank} Zui Zhong Jue Ding ] Jing Gao : Ben Ying Ji Xu Xun Lian , Dan Wei Zhao Dao Jian Cha Dian 。Jiang Cong Tou Kai Shi Xun Lian 。")
        else:
            print(f"[Rank {rank} Zui Zhong Jue Ding ] Jiang Diao Yong trainer.fit(model) (Cong Tou Kai Shi Xun Lian )")
        trainer.fit(model)

if __name__ == '__main__':
    fire.Fire(train_from_folder)