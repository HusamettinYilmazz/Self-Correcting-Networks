from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
from scipy.io import savemat
import torch

from datasets.sbd_dataset import SBDDataset

class SBDChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'img').mkdir()
        (self.root / 'cls').mkdir()
        self.split = 'supervised\nauxiliary\n'
        (self.root / 'train_noval.txt').write_text(self.split)
        self.mask = np.array([[0, 1, 255], [20, 2, 0]], dtype=np.uint8)
        self.image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
        for name in ['supervised', 'auxiliary']:
            Image.fromarray(self.image).save(self.root / 'img' / f'{name}.jpg')
            savemat(self.root / 'cls' / f'{name}.mat', {'GTcls': {'Segmentation': self.mask}})

    def test_excludes_supervised_ids_without_changing_split(self):
        dataset = SBDDataset(self.root, exclude_ids=['supervised', 'missing'])
        self.assertEqual(dataset.ids, ['auxiliary'])
        self.assertEqual((self.root / 'train_noval.txt').read_text(), self.split)
        self.assertEqual(SBDDataset(self.root).ids, ['supervised', 'auxiliary'])

    def test_no_transform_preserves_class_ids(self):
        image, target = SBDDataset(self.root)[0]
        self.assertEqual(image.shape, (2, 3, 3))
        self.assertEqual(target.dtype, torch.long)
        torch.testing.assert_close(target, torch.tensor(self.mask, dtype=torch.long))

    def test_box_mode_preserves_separate_instances_and_fills_rectangles(self):
        annotations = self.root / 'Annotations'
        annotations.mkdir()
        Image.fromarray(np.zeros((2, 5, 3), dtype=np.uint8)).save(self.root / 'img/supervised.jpg')
        (annotations / 'supervised.xml').write_text('''<annotation>
            <object><name>aeroplane</name><bndbox><xmin>0</xmin><ymin>0</ymin><xmax>2</xmax><ymax>2</ymax></bndbox></object>
            <object><name>aeroplane</name><bndbox><xmin>4</xmin><ymin>0</ymin><xmax>5</xmax><ymax>2</ymax></bndbox></object>
            </annotation>''')
        # The uploaded dataset has no inst folder; box mode must not read dense labels.
        (self.root / 'cls/supervised.mat').unlink()
        _, boxes = SBDDataset(self.root, return_boxes=True, exclude_ids=['auxiliary'],
                               box_annotations_path=annotations)[0]
        self.assertEqual(tuple(boxes.shape), (21, 2, 5))
        self.assertEqual(boxes.dtype, torch.float32)
        torch.testing.assert_close(boxes[1], torch.tensor([[1, 1, 0, 0, 1], [1, 1, 0, 0, 1.]]))
        self.assertEqual(boxes[0].sum().item(), 0)  # Matches existing VOC box encoding.

    def test_joint_transform_preserves_alignment_and_class_ids(self):
        def flip_to_tensor(*, image, mask):
            return {'image': torch.from_numpy(image[:, ::-1].copy()).permute(2, 0, 1),
                    'mask': torch.from_numpy(mask[:, ::-1].copy())}
        image, target = SBDDataset(self.root, transform=flip_to_tensor)[0]
        self.assertEqual(image.shape, (3, 2, 3))
        self.assertEqual(target.dtype, torch.long)
        torch.testing.assert_close(target, torch.tensor(self.mask[:, ::-1].copy(), dtype=torch.long))

if __name__ == '__main__':
    unittest.main()
