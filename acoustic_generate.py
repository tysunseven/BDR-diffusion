import torch
import numpy as np
# 更改: 导入 AcousticDiffusionModel
from network.model_trainer import AcousticDiffusionModel
from utils.utils import str2bool, ensure_directory
from utils.utils import num_to_groups
import argparse
import os
from tqdm import tqdm
from PIL import Image
import json as js

# 更改: 重命名函数并移除 bdr 参数
def generate_transmission(
    model_path: str,
    output_path: str = "./outputs",
    ema: bool = True,
    num_generate: int = 36,
    start_index: int = 0,
    steps: int = 50,
    truncated_time: float = 0.0,
    json_path="",
    # 移除: bdr_path 和 bdr_type
):
    model_name, model_id = model_path.split('/')[-2], model_path.split('/')[-1]
    
    # 更改: 加载 AcousticDiffusionModel
    discrete_diffusion = AcousticDiffusionModel.load_from_checkpoint(model_path).cuda()
    
    # 更改: 更新 postfix
    postfix = f"{model_name}_{model_id}_{ema}_{steps}_{truncated_time}_transmission"
    root_dir = os.path.join(output_path, postfix)


    # load json file
    with open(json_path,"r") as f:
            data=js.load(f)
    c1=np.array(data["C1"])
    
    # 更改: 创建 2-DOF 的 ctensor
    ctensor=np.zeros((2,c1.shape[0]))
    ctensor[0]=c1
    ctensor[1]=np.array(data["C2"])
    # 移除: ctensor[2]=np.array(data["C3"])

    # 移除: 所有 BDR 加载逻辑
    # mybdr=np.zeros((16,1,128,128))
    # ... (相关代码已删除) ...

    ensure_directory(root_dir)
    batches = num_to_groups(ctensor.shape[1], 16) #
    generator = discrete_diffusion.ema_model if ema else discrete_diffusion.model
    index = start_index

    gathered_samples=[]
    j=0
    for batch in batches:
        # 更改: 调用 sample_transmission 并移除 mybdr
        res_tensor = generator.sample_transmission(
            batch_size=batch, steps=steps, truncated_index=truncated_time, C=ctensor[:,j:j+batch].T)
        gathered_samples.extend(post_process(res_tensor).cpu().numpy())
        j+=16
    save_as_npz(gathered_samples,output_path)


def post_process(res_tensor):
    res_tensor = ((res_tensor+1)*127.5).clamp(0,255).to(torch.uint8)
    res_tensor = res_tensor.permute(0,2,3,1)
    res_tensor = res_tensor.contiguous()
    return res_tensor

def save_as_npz(gathered_samples,path):
    arr = np.array(gathered_samples)
    np.savez(path,arr)

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='generate something')
    
    # 更改: 更新默认方法和帮助信息
    parser.add_argument("--generate_method", type=str, default='generate_transmission',
                        help="please choose : 'generate_transmission' \n  \n \ ")

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="./outputs")
    parser.add_argument("--ema", type=str2bool, default=True)
    parser.add_argument("--num_generate", type=int, default=16)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--truncated_time", type=float, default=0.0)
    parser.add_argument("--data_class", type=str, default="microstructure")
    parser.add_argument("--text_w", type=float, default=1.0)
    parser.add_argument("--image_path", type=str, default="test.png")
    parser.add_argument("--image_name", type=str2bool, default=False)
    parser.add_argument("--elevation", type=float, default=0.)
    parser.add_argument("--kernel_size", type=float, default=4.)
    parser.add_argument("--verbose", type=str2bool, default=False)
    # json file of transmission coefficients
    parser.add_argument("--json_path", type=str, default="")
    
    # 移除: bdr_path 和 bdr_type 参数
    # parser.add_argument("--bdr_path", type=str, default="")
    # parser.add_argument("--bdr_type",type=int,default=2)

    args = parser.parse_args()
    method = (args.generate_method).lower()
    ensure_directory(args.output_path)

    # 更改: 调用新的 generate_transmission 函数
    if method == "generate_transmission":
        generate_transmission(model_path=args.model_path, num_generate=args.num_generate,
                               output_path=args.output_path, ema=args.ema, start_index=args.start_index, steps=args.steps,
                               truncated_time=args.truncated_time,json_path=args.json_path)
    else:
        raise NotImplementedError