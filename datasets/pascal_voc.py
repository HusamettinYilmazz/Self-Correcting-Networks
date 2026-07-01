import os 
import sys
import random
random.seed(42)

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

import xml.etree.ElementTree as ET

from utils.boxinfo import BoxInfo

class_map = {
    "background": 0,
    "aeroplane": 1, "bicycle": 2, "bird": 3, "boat": 4,
    "bottle": 5, "bus": 6, "car": 7, "cat": 8,
    "chair": 9, "cow": 10, "diningtable": 11, "dog": 12,
    "horse": 13, "motorbike": 14, "person": 15,
    "pottedplant": 16, "sheep": 17, "sofa": 18,
    "train": 19, "tvmonitor": 20
}

class VOCDataset(Dataset):
    def __init__(self, data_path, data_type="train", is_sup=True, split_ratio=1.0, transform=None):
        super().__init__()
        
        self.data_path = data_path
        self.data_type = data_type
        self.is_sup = is_sup
        self.split_ratio = split_ratio
        self.transform = transform
        self.data, self.bbox = self._load_data(self.data_type)

        rng = random.Random(42)
        indices = list(range(len(self.data)))
        rng.shuffle(indices)

        split = int(self.split_ratio * len(indices))

        if self.is_sup:
            selected_idx = indices[:split]
        else:
            selected_idx = indices[split:]

        self.data = [self.data[i] for i in selected_idx]
        self.bbox = {k: self.bbox[k] for k in self.data if k in self.bbox}

    def _load_data(self, data_type):
        file_path = os.path.join(self.data_path, "ImageSets", "Segmentation", f"{data_type}.txt")
        annot_file_path = os.path.join(self.data_path, "Annotations")

        data = []
        bbox = {}
        with open(file_path, 'r') as f:
            for line in f.readlines():
                line = line.strip() 
                data.append(line)
                bbox[line] = self._parse_xml(os.path.join(annot_file_path, f"{line}.xml"))

        return data, bbox

    def _parse_xml(self, xml_path): ## be sure of its correctness and use it inside _load_data
        """
        1. Build scheme for bbox
        2. Be sure it parsed properly
        3. use it inside _load_data return: bbox with the path of the image
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()
        img_name = root.find("filename").text[:-4]

        boxes = []
        for obj in root.findall("object"):
            cls = obj.find("name").text

            # skip difficult objects if you want
            difficult = obj.find("difficult")
            if difficult is not None and int(difficult.text) == 1:
                continue

            bndbox = obj.find("bndbox")

            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)

            bbox = BoxInfo(cls, xmin, ymin, xmax, ymax, img_name)

            boxes.append(bbox)

        return boxes

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_num = self.data[idx]

        img_path = os.path.join(self.data_path, "JPEGImages", f"{img_num}.jpg")
        mask_path = os.path.join(self.data_path, "SegmentationClass", f"{img_num}.png")

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))

        H, W = mask.shape

        num_classes = 21

        # (H, W, C)
        weak_mask = np.zeros((H, W, num_classes), dtype=np.uint8)

        for box in self.bbox[img_num]:
            cls_name = box.label

            if cls_name not in class_map:
                continue

            cls_id = class_map[cls_name]

            xmin, ymin, xmax, ymax = box.box

            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(W, xmax)
            ymax = min(H, ymax)

            weak_mask[ymin:ymax, xmin:xmax, cls_id] = 1

        if self.transform:
            transformed = self.transform(
                image=image,
                masks=[mask] + [weak_mask[..., c] for c in range(num_classes)]
            )

            image = transformed["image"]

            transformed_masks = transformed["masks"]

            mask = transformed_masks[0]

            weak_mask = np.stack(transformed_masks[1:], axis=0)

        else:
            weak_mask = weak_mask.transpose(2, 0, 1)

        weak_mask = torch.tensor(weak_mask, dtype=torch.float32)

        if self.is_sup:
            return image, weak_mask, mask

        return image, weak_mask
