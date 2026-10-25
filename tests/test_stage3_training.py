import copy
import logging
import tempfile
import unittest
from pathlib import Path

import torch

from train.train_3rd_stage import train_primary_model_epoch, validate_primary_model, initialize_stage3
from utils.dice_loss import DiceLoss
from utils.helper import Config, save_checkpoint
from utils.lr_scheduler import Stage3PolynomialLR


class Ancillary(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(2)

    def forward(self, images, boxes):
        return self.bn(boxes.float())


class Corrector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Conv2d(4, 2, 1)
        self.seen_modes = []

    def forward(self, primary, ancillary):
        self.seen_modes.append((self.training, torch.is_grad_enabled()))
        return self.head(torch.cat([primary, ancillary], dim=1))


class Stage3TrainingTests(unittest.TestCase):
    def setup_training(self, batches=1):
        torch.manual_seed(42)
        models = {'primary': torch.nn.Conv2d(1, 2, 1),
                  'ancillary': Ancillary(), 'correcting': Corrector()}
        optimizers = {name: torch.optim.SGD(models[name].parameters(), lr=0.1)
                      for name in ('primary', 'correcting')}
        schedulers = {name: torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
                      for name, opt in optimizers.items()}
        image = torch.ones(1, 1, 2, 2)
        target = torch.ones(1, 2, 2, dtype=torch.long)
        boxes = torch.cat([torch.zeros_like(image), torch.ones_like(image)], dim=1)
        kwargs = dict(epoch=1, data_loaders={'f_loader': [(image, boxes, target)],
                     'w_loader': [(image, boxes)] * batches}, device='cpu', models=models,
                     optimizers=optimizers, schedulers=schedulers,
                     loss_funcs={'ce_loss': torch.nn.CrossEntropyLoss(), 'dice_loss': DiceLoss()},
                     accum_steps=1, logger=logging.getLogger('stage3_test'))
        return kwargs

    def test_self_correction_updates_both_models_but_not_ancillary(self):
        kwargs = self.setup_training()
        before = {name: copy.deepcopy(model.state_dict()) for name, model in kwargs['models'].items()}
        train_primary_model_epoch(**kwargs)
        for name in ('primary', 'correcting'):
            self.assertTrue(any(not torch.equal(value, kwargs['models'][name].state_dict()[key])
                                for key, value in before[name].items()))
            self.assertEqual(kwargs['schedulers'][name].last_epoch, 1)
        for key, value in before['ancillary'].items():
            torch.testing.assert_close(value, kwargs['models']['ancillary'].state_dict()[key])
        self.assertFalse(kwargs['models']['ancillary'].training)
        self.assertTrue(all(p.grad is None for p in kwargs['models']['ancillary'].parameters()))
        self.assertEqual(kwargs['models']['correcting'].seen_modes, [(False, False), (True, True)])

    def test_stage2_initialization_and_bundled_stage3_resume(self):
        kwargs = self.setup_training()
        with tempfile.TemporaryDirectory() as directory:
            paths = {}
            for name in ('primary', 'correcting', 'ancillary'):
                optimizer = kwargs['optimizers'].get(name) or torch.optim.SGD(kwargs['models'][name].parameters(), lr=0.1)
                scheduler = kwargs['schedulers'].get(name) or torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
                kwargs['optimizers'][name] = optimizer
                kwargs['schedulers'][name] = scheduler
                save_checkpoint(9, kwargs['models'][name], optimizer, scheduler, 0.1,
                                None, None, None, None, directory, model_name=name)
                paths[name] = Path(directory) / f'epoch9_{name}.pth'
            start = initialize_stage3(paths, kwargs['models'], kwargs['optimizers'], kwargs['schedulers'],
                                      'cpu', resume=False, mode='self_correction')
            self.assertEqual(start, 1)
            for name in ('primary', 'correcting'):
                kwargs['schedulers'][name] = Stage3PolynomialLR(kwargs['optimizers'][name], 3, 1)
                self.assertEqual(kwargs['schedulers'][name].last_epoch, 0)
            train_primary_model_epoch(**kwargs)
            save_checkpoint(
                1, kwargs['models']['primary'], kwargs['optimizers']['primary'], kwargs['schedulers']['primary'],
                kwargs['optimizers']['primary'].param_groups[0]['lr'], None,
                Config({'training': {'stage3_mode': 'self_correction'}}), None, None, directory,
                filename='latest_primary.pth', companions={'correcting': (kwargs['models']['correcting'],
                         kwargs['optimizers']['correcting'], kwargs['schedulers']['correcting'])},
            )
            resumed = self.setup_training()
            resumed['optimizers']['ancillary'] = torch.optim.SGD(resumed['models']['ancillary'].parameters(), lr=0.1)
            resumed['schedulers']['ancillary'] = torch.optim.lr_scheduler.LambdaLR(resumed['optimizers']['ancillary'], lambda _: 1.)
            for name in ('primary', 'correcting'):
                resumed['schedulers'][name] = Stage3PolynomialLR(resumed['optimizers'][name], 3, 1)
            start = initialize_stage3({'primary': Path(directory) / 'latest_primary.pth',
                                      'ancillary': paths['ancillary']}, resumed['models'], resumed['optimizers'],
                                      resumed['schedulers'], 'cpu', resume=True, mode='self_correction')
            self.assertEqual(start, 2)
            for name in ('primary', 'correcting'):
                self.assertEqual(resumed['schedulers'][name].state_dict(), kwargs['schedulers'][name].state_dict())
                for key, value in kwargs['models'][name].state_dict().items():
                    torch.testing.assert_close(value, resumed['models'][name].state_dict()[key])

    def test_corrected_targets_change_primary_update(self):
        results = []
        for winning_class in (0, 1):
            kwargs = self.setup_training()
            with torch.no_grad():
                head = kwargs['models']['correcting'].head
                head.weight.zero_()
                head.bias.fill_(-10)
                head.bias[winning_class] = 10
            train_primary_model_epoch(**kwargs)
            results.append(kwargs['models']['primary'].weight.detach().clone())
        self.assertFalse(torch.equal(*results))

    def test_partial_accumulation_has_full_strength(self):
        # With LR=0 in the first group, all three losses see identical weights.
        # The last one-batch group must equal a standalone one-batch update.
        full = self.setup_training(batches=3)
        single = self.setup_training()
        for kwargs in (full, single):
            kwargs['mode'] = 'supervised'
            image, _, target = kwargs['data_loaders']['f_loader'][0]
            kwargs['data_loaders']['w_loader'] = [(image, target)] * (3 if kwargs is full else 1)
        full['accum_steps'] = 2
        opt = full['optimizers']['primary']
        opt.param_groups[0]['lr'] = 0.0
        # A scheduler sets the LR for the second optimizer update.
        class EnableLR:
            def step(self):
                opt.param_groups[0]['lr'] = 0.1
        full['schedulers']['primary'] = EnableLR()
        train_primary_model_epoch(**full)
        train_primary_model_epoch(**single)
        for actual, expected in zip(full['models']['primary'].parameters(),
                                    single['models']['primary'].parameters()):
            torch.testing.assert_close(actual, expected)

    def test_validation_reports_all_classes_and_foreground_separately(self):
        model = torch.nn.Conv2d(1, 2, 1)
        with torch.no_grad():
            model.weight.zero_()
            model.bias.copy_(torch.tensor([10., -10.]))
        metrics = validate_primary_model(
            epoch=1, data_loader=[(torch.ones(1, 1, 2, 2), None,
                                   torch.zeros(1, 2, 2, dtype=torch.long))],
            device='cpu', models={'primary': model},
            loss_funcs={'ce_loss': torch.nn.CrossEntropyLoss()},
            class_names=['background', 'foreground'], logger=logging.getLogger('stage3_test'),
        )
        self.assertAlmostEqual(metrics['primary_mIoU'], 0.5)
        self.assertEqual(metrics['primary_foreground_mIoU'], 0.0)


if __name__ == '__main__':
    unittest.main()
