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
        super().__init__() # 初始化父类
        self.resolution = resolution # 承载的是训练时配置的image_size，但实际上并未使用
        self.data_folder=data_folder # 数据集路径，这个肯定是要的
        # os.path.join(data_folder,"img") 是知道了img这个data_folder的子文件夹里存放的是图片文件
        # _list_image_files_recursively返回一个列表，列表里存放的是所有图片文件的路径
        self.images = _list_image_files_recursively(os.path.join(data_folder,"img"))
        # os.path.join(data_folder,"count.json")是知道了这个文件里存储的是弹性张量的数据
        # "r"是 open() 函数的第二个参数，表示以只读模式打开文件
        # as f 用于将 open() 函数返回的文件对象（也叫文件句柄）赋值给一个变量f
        with open(os.path.join(data_folder,"count.json"),"r") as f:
            # js.load(f) 是将文件对象 f 中的 JSON 数据读取并解析为 Python 对象
            # 顶层结构是一个JSON 对象 ( {} )，会被解析为 Python 字典
            # 键 '0' 和 '1' 是 Python 字符串 ( str )
            # 与每个键对应的值是 Python 列表 ( list )
            # 列表中的元素是 Python 浮点数 ( float )
            self.elatsic_tensor=js.load(f)

    # DataLoader 会在后台自动地、隐式地调用这个方法
    # 知道数据集中有多少个样本，以便确定如何采样和划分批次
    def __len__(self):
        return len(self.images)

    # getitem 方法用于根据索引 idx 获取数据集中的一个样本
    # DataLoader 会在后台自动地、隐式地调用这个方法
    # 然后把返回的样本数据打包成一个 batch
    def __getitem__(self, idx):
        # 之前说self.images里存放的是图片文件的路径
        # 现在 path 就是第 idx 个图片文件的路径
        path = self.images[idx]
        # os.path.split(path) 会将路径 path 分割成目录和文件名两部分
        # 比如 os.path.split("/home/data/img/0.png") 会返回 ("/home/data/img", "0.png")
        # 然后 [1] 会取出文件名部分 "0.png"
        # os.path.splitext(...) 会将文件名分割成文件名前缀和扩展名两部分
        # 比如 os.path.splitext("0.png") 会返回 ("0", ".png")
        # front, _ = ... 它会把右侧元组 ('0', '.png') 中的元素，按顺序依次赋值给左侧的变量
        # 下划线 _ 是一个约定俗成的**“占位符”或“丢弃”**变量，表示我们不关心这个值
        front, _=os.path.splitext(os.path.split(path)[1])
        # rb 是 open() 函数的第二个参数，表示以二进制只读模式打开文件
        with open(path, "rb") as f:
            # Image.open(...)是 PIL 库中用于打开图像文件的标准函数
            # Image.open(f)返回一个代表图像的 PIL 内部对象
            img = Image.open(f)
            # img.load()：调用这个方法会强制 PIL 库读取并解码完整的图像数据（所有的像素点），并将其加载到内存中。
            img.load()
        # img.convert("L")返回一个新的 PIL Image 对象
        # 无论原始图像是什么格式，这个新对象都将被转换为 8 位单通道的灰度图像
        # 每个像素只用一个 0 到 255 之间的值来表示其亮度
        # np.array(...)：将 PIL Image 对象转换为 NumPy 数组
        # 如果图像的尺寸是 (H, W)，这个函数会返回一个形状为 (H, W) 的 numpy 数组
        # 数组中的每个元素都是一个 0 到 255 之间的整数（uint8）
        img = np.array(img.convert("L"))
        # img.astype(np.float32)将 img 数组中的所有整数转换为浮点数
        # 除以 127.5 并减去 1，将像素值从 [0, 255] 映射到 [-1, 1]
        # 这种归一化处理有助于神经网络更稳定、更高效地训练
        img = img.astype(np.float32) / 127.5 - 1
        # bdr对我没有什么借鉴价值
        bdr=np.ones_like(img)
        idx=np.where(img[0,:]==-1)[0]
        bdr[idx,:]=-1
        bdr[:,idx]=-1
        # img[np.newaxis,:] 是一种 numpy 索引语法，它的核心作用是在数组的开头添加一个新维度
        # 假设 img 数组的形状是 (64, 64)（代表一个 64x64 的图像）
        # 那么 img[np.newaxis, :] 的结果将是一个形状为 (1, 64, 64) 的新数组
        return {"img":img[np.newaxis,:],
                # self.elatsic_tensor[front]]是 Python 字典的键查找操作
                # self.elatsic_tensor：我们知道这是一个字典
                # front：这是我们之前从文件名中提取的键（例如 '0'）
                # self.elatsic_tensor['0'] 会返回与键 '0' 对应的值，也就是那个列表
                # ...[:-1] 切片操作会返回列表中除了最后一个元素之外的所有元素
                # np.array(..., dtype=np.float32)用于将输入的列表转换为 numpy 数组
                # dtype=np.float32 指定了数组中元素的数据类型为 32 位浮点数
                "cond":np.array([v for v in self.elatsic_tensor[front]][:-1],dtype=np.float32),
                "bdr":bdr[np.newaxis,:],
        }
        # 返回的数据类型是一个 Python 字典，包含三个键值对
        # DataLoader 准备好的 batch 字典作为参数传递给 training_step 方法
        # 可以看到 model_trainer.py 中的 training_step 方法接受一个参数 batch
        # 在 training_step 中使用 batch["img"]、batch["cond"] 和 batch["bdr"] 这样的语法来获取数据

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
        data_folder=None, # 此参数现在指向包含 .npy 文件的目录
        structures_path=None,  # 新增: 直接指定结构文件路径
        properties_path=None   # 新增: 直接指定属性文件路径
    ):
        super().__init__()
        self.resolution = resolution
        
        # 优先使用指定的文件路径，如果没有指定，则回退到原来的 data_folder 拼接方式
        if structures_path is not None and properties_path is not None:
            struct_p = structures_path
            prop_p = properties_path
        elif data_folder is not None:
            struct_p = os.path.join(data_folder, "surrogate_structures.npy")
            prop_p = os.path.join(data_folder, "surrogate_properties.npy")
        else:
            raise ValueError("Must provide either (structures_path and properties_path) or data_folder")
        
        self.structures = np.load(struct_p)
        self.properties = np.load(prop_p)

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