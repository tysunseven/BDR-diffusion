#!/bin/bash
export RESULT_FOLDER="/root/autodl-tmp/bdr_results"
export RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
python3 train.py --run_timestamp $RUN_TIMESTAMP --results_folder $RESULT_FOLDER --data_class microstructure --name model --batch_size 256 --continue_training False --image_size 128 --training_epoch 200 --ema_rate 0.999 --base_channels 32 --save_last True --save_every_epoch 50 --with_attention True  --split_dataset False  --lr 1e-4 --optimizier adamw --img_folder /root/autodl-fs
