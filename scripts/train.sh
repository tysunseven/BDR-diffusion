#!/bin/bash
export RESULT_FOLDER="/root/autodl-tmp/bdr_results"
python3 train.py --results_folder $RESULT_FOLDER --data_class microstructure --name model --batch_size 8 --new True --continue_training False --image_size 128 --training_epoch 2000 --ema_rate 0.999 --base_channels 32 --save_last False --save_every_epoch 500 --with_attention True  --split_dataset False  --lr 1e-4 --optimizier adamw --img_folder /root/autodl-fs
