#!/bin/bash
export RESULT_FOLDER="/root/autodl-tmp/bdr_results"

# --- 配置区 ---
# 设置为 'true' 来继续训练, 'false' 来开始新训练
CONTINUE_TRAINING=true

# 仅在 CONTINUE_TRAINING=true 时使用:
# 在这里填入您想要继续训练的那个文件夹的时间戳
# 例如: 20251115142819
LOAD_TIMESTAMP_ID="20251115170107" 
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

python3 train.py --run_timestamp $RUN_TIMESTAMP --results_folder $RESULT_FOLDER --data_class microstructure --name model --batch_size 256 --continue_training $CONTINUE_TRAINING --image_size 128 --training_epoch 200 --ema_rate 0.999 --base_channels 32 --save_last True --save_every_epoch 50 --with_attention True  --split_dataset False  --lr 1e-4 --optimizier adamw --img_folder /root/autodl-fs
