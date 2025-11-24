#!/bin/bash

python3 generate.py --step 50 --generate_method generate_based_on_bdr_json --model_path /root/autodl-tmp/bdr_results/model_20251115170107/epoch=3999.ckpt --output_path /root/autodl-tmp/bdr_results/model_20251115170107/output --bdr_path /root/autodl-fs/img --json_path /root/autodl-fs/countT.json 