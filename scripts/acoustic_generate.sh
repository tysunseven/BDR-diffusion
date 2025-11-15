#!/bin/bash

# 更改:
# 1. 调用 acoustic_generate.py
# 2. 将 --step 修正为 --steps
# 3. 将 --generate_method 更改为 generate_transmission
# 4. 移除了 --bdr_path 和 --bdr_type (原脚本中未指定 --bdr_type)
# 5. 假设 --json_path 指向新的透射系数文件 (例如 transmission.json)

/usr/bin/python3 acoustic_generate.py \
    --steps 50 \
    --generate_method generate_transmission \
    --model_path /data1/fjx/bdr-models/model8/model/epoch=1999.ckpt \
    --output_path /data1/fjx/bdr-models/model8/output_target/output \
    --json_path /home/fjx/bdr62/transmission.json