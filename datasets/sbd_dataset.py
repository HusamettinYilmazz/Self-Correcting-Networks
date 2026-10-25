import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import scipy.io as sio


class SBDDataset(Dataset):
    def __init__(self, data_path, transform=None, exclude_ids=None):
        super().__init__()

        self.data_path = data_path
        self.transform = transform

        self.img_dir = os.path.join(data_path, "img")
        self.ann_dir = os.path.join(data_path, "cls")

        self.ids = self._load_data()
        if exclude_ids is not None:
            excluded = set(exclude_ids)
            self.ids = [img_id for img_id in self.ids if img_id not in excluded]

    def _load_data(self):
        path = os.path.join(self.data_path, "train_noval.txt")
        with open(path, "r") as f:
            return [line.strip() for line in f]

    def _load_mask(self, img_id):
        mat_path = os.path.join(self.ann_dir, f"{img_id}.mat")
        data = sio.loadmat(mat_path)
        
        mask = data["GTcls"][0]["Segmentation"][0]

        return mask.astype(np.int64)

    def _mask_to_onehot(self, mask, num_classes=21):
        H, W = mask.shape
        onehot = np.zeros((num_classes, H, W), dtype=np.uint8)

        # valid = mask != 255
        for c in range(num_classes):
            onehot[c] = (mask == c) #& valid

        return onehot

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]

        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
        image = np.array(Image.open(img_path).convert("RGB"))

        target_mask = self._load_mask(img_id)  # (H, W), class IDs including ignore label 255

        if self.transform:
            transformed = self.transform(
                image=image,
                mask=target_mask
            )

            image = transformed["image"]
            target_mask = transformed["mask"]

        target_mask = torch.as_tensor(target_mask, dtype=torch.long)
        return image, target_mask
