import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import scipy.io as sio
import xml.etree.ElementTree as ET

from .pascal_voc import class_map


class SBDDataset(Dataset):
    def __init__(self, data_path, transform=None, exclude_ids=None, return_boxes=False,
                 num_classes=21, box_annotations_path=None):
        super().__init__()

        self.data_path = data_path
        self.transform = transform
        self.return_boxes = return_boxes
        self.num_classes = num_classes

        self.img_dir = os.path.join(data_path, "img")
        self.ann_dir = os.path.join(data_path, "cls")
        self.box_annotations_path = box_annotations_path

        self.ids = self._load_data()
        if exclude_ids is not None:
            excluded = set(exclude_ids)
            self.ids = [img_id for img_id in self.ids if img_id not in excluded]
        if self.return_boxes:
            if self.box_annotations_path is None:
                raise ValueError('Box mode requires the VOC Annotations directory via box_annotations_path.')
            for img_id in self.ids:
                path = os.path.join(self.box_annotations_path, f'{img_id}.xml')
                if not os.path.isfile(path):
                    raise FileNotFoundError(f'Missing VOC bounding-box annotation: {path}')

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

    def _load_boxes(self, img_id, image_shape):
        # Kaggle's SBD upload contains cls/ and img/, not inst/. Actual object
        # boxes come from the VOC dataset already mounted for supervised training.
        root = ET.parse(os.path.join(self.box_annotations_path, f'{img_id}.xml')).getroot()
        height, width = image_shape
        boxes = np.zeros((self.num_classes, height, width), dtype=np.float32)
        for obj in root.findall('object'):
            class_id = class_map.get(obj.findtext('name'))
            if class_id is None or not 0 < class_id < self.num_classes:
                continue
            if obj.findtext('difficult', '0') == '1':
                continue
            bounds = obj.find('bndbox')
            xmin = max(0, int(bounds.findtext('xmin')))
            ymin = max(0, int(bounds.findtext('ymin')))
            xmax = min(width, int(bounds.findtext('xmax')))
            ymax = min(height, int(bounds.findtext('ymax')))
            boxes[class_id, ymin:ymax, xmin:xmax] = 1
        # Match the existing VOC box inputs and trained ancillary checkpoints:
        # background channel zero, with separate rectangles per foreground instance.
        return boxes

    def __getitem__(self, idx):
        img_id = self.ids[idx]

        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.return_boxes:
            boxes = self._load_boxes(img_id, image.shape[:2])
            if self.transform:
                transformed = self.transform(image=image, masks=list(boxes))
                image = transformed['image']
                boxes = torch.stack([torch.as_tensor(mask) for mask in transformed['masks']])
            return image, torch.as_tensor(boxes, dtype=torch.float32)

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
