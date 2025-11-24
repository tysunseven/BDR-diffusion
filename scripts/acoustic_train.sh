#!/bin/bash
export RESULT_FOLDER="/root/autodl-tmp/acoustic_results"

# --- 配置区 ---
# 设置为 'true' 来继续训练, 'false' 来开始新训练
CONTINUE_TRAINING=false

# 仅在 CONTINUE_TRAINING=true 时使用:
# 在这里填入您想要继续训练的那个文件夹的时间戳
# 例如: 20251115142819
LOAD_TIMESTAMP_ID="20251120082359" 
# LOAD_TIMESTAMP_ID="20251120082359" 这个是目前训练了8000epoch的模型 

# --- 数据集路径配置 (新增) ---
# 训练集文件
TRAIN_STRUCT="/root/autodl-fs/acoustic_dataset2/trainset2_structures.npy"
TRAIN_PROP="/root/autodl-fs/acoustic_dataset2/trainset2_properties.npy"

# 验证集文件 (请修改为你实际的路径)
VAL_STRUCT="/root/autodl-fs/acoustic_dataset2/valiset2_structures.npy"
VAL_PROP="/root/autodl-fs/acoustic_dataset2/valiset2_properties.npy"
# ---------------------------

# --- 结束配置 ---

# --- DDP 安全逻辑 (自动) ---
if [ "$CONTINUE_TRAINING" = "true" ]; then
    echo "--- 模式: 继续训练 ---"
    # 使用您指定的已存在的时间戳
    export RUN_TIMESTAMP=$LOAD_TIMESTAMP_ID
else
    echo "--- 模式: 开始新训练 ---"
    # 生成一个新的唯一时间戳
    export RUN_TIMESTAMP=$(date +%Y%m%d%H%M%S)
fi
echo "使用的时间戳 ID: $RUN_TIMESTAMP"
# --- 结束逻辑 ---

# 更改: 调用新的训练脚本 acoustic_train.py
python3 acoustic_train.py --run_timestamp $RUN_TIMESTAMP \
                        --results_folder $RESULT_FOLDER \
                        --data_class microstructure \
                        --name model \
                        --batch_size 256 \
                        --continue_training $CONTINUE_TRAINING \
                        --image_size 8 \
                        --training_epoch 8020 \
                        --ema_rate 0.999 \
                        --base_channels 32 \
                        --save_last True \
                        --save_every_epoch 50 \
                        --with_attention True  \
                        --split_dataset False  \
                        --lr 1e-4 \
                        --optimizier adamw \
                        # --img_folder /root/autodl-fs/acoustic_dataset1
                        --train_structures_path "$TRAIN_STRUCT" \
                        --train_properties_path "$TRAIN_PROP" \
                        --val_structures_path "$VAL_STRUCT" \
                        --val_properties_path "$VAL_PROP"