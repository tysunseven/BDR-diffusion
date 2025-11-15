import torch
import numpy as np
import os
import json as js
from PIL import Image

def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(os.listdir(data_dir)):
        full_path = os.path.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif os.path.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results

class ImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        resolution,
        data_folder,
    ):
        super().__init__()
        self.resolution = resolution
        self.data_folder=data_folder
        self.images = _list_image_files_recursively(os.path.join(data_folder,"img"))
        with open(os.path.join(data_folder,"count.json"),"r") as f:
            self.elatsic_tensor=js.load(f)


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = self.images[idx]
        front, _=os.path.splitext(os.path.split(path)[1])
        with open(path, "rb") as f:
            img = Image.open(f)
            img.load()
        img = np.array(img.convert("L"))
        img = img.astype(np.float32) / 127.5 - 1
        bdr=np.ones_like(img)
        idx=np.where(img[0,:]==-1)[0]
        bdr[idx,:]=-1
        bdr[:,idx]=-1
        return {"img":img[np.newaxis,:],
                "cond":np.array([v for v in self.elatsic_tensor[front]][:-1],dtype=np.float32),
                "bdr":bdr[np.newaxis,:],
        } 

class AcousticDataset(torch.utils.data.Dataset):
    """
    用于加载声学超材料数据集的 Dataset 类。
    - 更改: 从两个 .npy 文件加载数据 (模仿 Acoustic-Metamaterial-Generator)。
    - 条件(cond): 加载 2-DOF 的透射系数。
    - 移除: 不加载或处理 BDR (边界条件) 数据。
    """
    def __init__(
        self,
        resolution,  # 此参数保留以匹配 Lightning 模块的签名，但不再需要
        data_folder, # 此参数现在指向包含 .npy 文件的目录
    ):
        super().__init__()
        self.resolution = resolution
        self.data_folder = data_folder
        
        # 更改: 加载 .npy 文件，而不是遍历 img 文件夹
        # 假设 .npy 文件位于 data_folder 中
        structures_path = os.path.join(self.data_folder, "surrogate_structures.npy")
        properties_path = os.path.join(self.data_folder, "surrogate_properties.npy")

        self.structures = np.load(structures_path)
        self.properties = np.load(properties_path)

        # 检查: 确保属性是 2-DOF (或至少 2-DOF)
        assert self.properties.shape[1] >= 2, "Properties .npy file must have at least 2 columns."

        # 归一化: Acoustic-Metamaterial-Generator 的数据是 [0, 1]
        # 将其转换为模型期望的 [-1, 1] 范围
        self.structures = (self.structures.astype(np.float32) * 2.0) - 1.0

    def __len__(self):
        # 更改: 返回 .npy 数组的长度
        return len(self.structures)

    def __getitem__(self, idx):
        # 更改: 从加载的 numpy 数组中获取数据
        img = self.structures[idx]
        # 假设前两列是所需的 2-DOF 条件
        cond = self.properties[idx][:2] 
        
        # 移除: 所有 BDR 相关的逻辑
        
        # 更改: 返回值
        return {
            "img": img[np.newaxis, :], # 保持 [B, C=1, H, W] 的形状
            "cond": cond.astype(np.float32)
        }