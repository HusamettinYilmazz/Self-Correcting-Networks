import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import scipy.io as sio


class SBDDataset(Dataset):
    def __init__(self, data_path, transform=None ):
        super().__init__()

        self.data_path = data_path
        self.transform = transform

        self.img_dir = os.path.join(data_path, "img")
        self.ann_dir = os.path.join(data_path, "cls")

        self.ids = self._load_data()

    def _load_data(self):
        path = os.path.join(self.data_path, "train_noval.txt")
        with open(path, "r") as f:
            return [line.strip() for line in f]

    def _load_mask(self, img_id):
        mat_path = os.path.join(self.ann_dir, f"{img_id}.mat")
        data = sio.loadmat(mat_path)
        
        mask = data["GTcls"][0]["Segmentation"][0]

        return mask.astype(np.int64)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]

        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
        image = np.array(Image.open(img_path).convert("RGB"))

        mask = self._load_mask(img_id)

        if self.transform:
            transformed = self.transform(
                image=image,
                masks=[mask]
            )

            image = transformed["image"]
            mask = transformed["masks"][0]

        image = torch.tensor(image).permute(2, 0, 1).float()
        mask = torch.tensor(mask, dtype=torch.long)

        return image, mask
