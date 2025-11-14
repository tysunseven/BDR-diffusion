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
