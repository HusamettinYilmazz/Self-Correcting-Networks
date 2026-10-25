import logging
import tempfile
import unittest
from pathlib import Path

import torch

from utils.helper import load_checkpoint, save_checkpoint
from utils.lr_scheduler import Stage3PolynomialLR
from utils.dice_loss import DiceLoss


class Stage3SchedulerTests(unittest.TestCase):
    def make_training(self, epochs=4, batches=5, accumulation=2):
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        scheduler = Stage3PolynomialLR(optimizer, epochs, batches, accumulation)
        return model, optimizer, scheduler

    def update(self, model, optimizer, scheduler):
        optimizer.zero_grad()
        model(torch.ones(1, 1)).sum().backward()
        optimizer.step()
        scheduler.step()
        return optimizer.param_groups[0]['lr']

    def save(self, directory, model, optimizer, scheduler, epoch):
        save_checkpoint(
            epoch, model, optimizer, scheduler, optimizer.param_groups[0]['lr'],
            None, None, None, None, directory, model_name='primary',
            filename='latest_primary.pth',
        )
        return Path(directory) / 'latest_primary.pth'

    def test_counts_partial_accumulation_and_clamps_at_zero(self):
        model, optimizer, scheduler = self.make_training(epochs=3)
        self.assertEqual(scheduler.updates_per_epoch, 3)
        self.assertEqual(scheduler.total_steps, 9)
        for step in range(1, 12):
            lr = self.update(model, optimizer, scheduler)
            self.assertAlmostEqual(lr, 0.1 * max(0, 1 - step / 9) ** 0.9)

    def test_stage3_epoch_updates_schedule_once_per_accumulated_weak_batch(self):
        from train.train_3rd_stage import train_primary_model_epoch
        model = torch.nn.Conv2d(1, 2, kernel_size=1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = Stage3PolynomialLR(optimizer, 1, 5, accum_steps=2)
        image = torch.ones(1, 1, 2, 2)
        target = torch.ones(1, 2, 2, dtype=torch.long)
        train_primary_model_epoch(
            epoch=1,
            data_loaders={'f_loader': [(image, None, target)],
                          'w_loader': [(image, target)] * 5},
            device='cpu', models={'primary': model},
            optimizers={'primary': optimizer},
            loss_funcs={'ce_loss': torch.nn.CrossEntropyLoss(), 'dice_loss': DiceLoss()},
            schedulers={'primary': scheduler}, accum_steps=2,
            logger=logging.getLogger('stage3_scheduler_test'),
            mode='supervised',
        )
        self.assertEqual(scheduler.last_epoch, 3)
        self.assertEqual(optimizer.param_groups[0]['lr'], 0.0)

    def test_multiple_sessions_match_uninterrupted_lr_and_momentum(self):
        full = self.make_training()
        expected = [self.update(*full) for _ in range(12)]
        current = self.make_training()
        actual = []
        with tempfile.TemporaryDirectory() as directory:
            for epoch in range(1, 5):
                actual.extend(self.update(*current) for _ in range(3))
                path = self.save(directory, *current, epoch)
                self.assertFalse(Path(str(path) + '.tmp').exists())
                current = self.make_training()
                state = load_checkpoint(path, *current, device='cpu')
                self.assertEqual(state['epoch'], epoch)
                self.assertEqual(current[2].last_epoch, epoch * 3)
                self.assertEqual(current[2].total_steps, 12)
        self.assertEqual(actual, expected)
        torch.testing.assert_close(current[0].weight, full[0].weight)
        actual_momentum = next(iter(current[1].state.values()))['momentum_buffer']
        expected_momentum = next(iter(full[1].state.values()))['momentum_buffer']
        torch.testing.assert_close(actual_momentum, expected_momentum)

    def test_legacy_checkpoint_preserves_lr_then_finishes_remaining_decay(self):
        model, optimizer, _ = self.make_training()
        legacy = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: (1 - step / 100) ** 0.9,
        )
        for _ in range(6):
            self.update(model, optimizer, legacy)
        saved_lr = optimizer.param_groups[0]['lr']
        with tempfile.TemporaryDirectory() as directory:
            path = self.save(directory, model, optimizer, legacy, epoch=2)
            resumed = self.make_training()
            with self.assertWarnsRegex(UserWarning, 'legacy'):
                load_checkpoint(path, *resumed, device='cpu')
            self.assertEqual(resumed[1].param_groups[0]['lr'], saved_lr)
            self.assertEqual(resumed[2].last_epoch, 6)
            self.assertEqual(resumed[2].total_steps, 12)
            for step in range(1, 7):
                lr = self.update(*resumed)
                self.assertAlmostEqual(lr, saved_lr * (1 - step / 6) ** 0.9)

    def test_changed_training_budget_rebases_without_lr_jump(self):
        current = self.make_training()
        for _ in range(6):
            self.update(*current)
        saved_lr = current[1].param_groups[0]['lr']
        with tempfile.TemporaryDirectory() as directory:
            path = self.save(directory, *current, epoch=2)
            resumed = self.make_training(epochs=5, batches=8)
            with self.assertWarnsRegex(UserWarning, 'changed'):
                load_checkpoint(path, *resumed, device='cpu')
            self.assertEqual(resumed[1].param_groups[0]['lr'], saved_lr)
            self.assertEqual(resumed[2].last_epoch, 8)
            self.assertEqual(resumed[2].total_steps, 20)
            self.assertAlmostEqual(self.update(*resumed), saved_lr * (1 - 1 / 12) ** 0.9)

    def test_weights_only_loading_keeps_new_optimizer_and_schedule(self):
        current = self.make_training()
        for _ in range(6):
            self.update(*current)
        with tempfile.TemporaryDirectory() as directory:
            path = self.save(directory, *current, epoch=2)
            fresh = self.make_training(epochs=6)
            load_checkpoint(path, *fresh, device='cpu', resume=False)
            torch.testing.assert_close(fresh[0].weight, current[0].weight)
            self.assertEqual(fresh[1].param_groups[0]['lr'], 0.1)
            self.assertEqual(len(fresh[1].state), 0)
            self.assertEqual(fresh[2].last_epoch, 0)
            self.assertEqual(fresh[2].total_steps, 18)

    def test_extending_exhausted_schedule_requires_fresh_finetuning(self):
        current = self.make_training(epochs=1)
        for _ in range(3):
            self.update(*current)
        with tempfile.TemporaryDirectory() as directory:
            path = self.save(directory, *current, epoch=1)
            resumed = self.make_training(epochs=2)
            with self.assertRaisesRegex(ValueError, 'zero'):
                load_checkpoint(path, *resumed, device='cpu')

    def test_one_atomic_checkpoint_restores_primary_and_corrector(self):
        primary, corrector = self.make_training(), self.make_training()
        for _ in range(6):
            self.update(*primary)
            self.update(*corrector)
        with tempfile.TemporaryDirectory() as directory:
            save_checkpoint(
                2, *primary, primary[1].param_groups[0]['lr'], None, None,
                None, None, directory, filename='latest_primary.pth',
                companions={'correcting': corrector},
            )
            new_primary, new_corrector = self.make_training(), self.make_training()
            state = load_checkpoint(Path(directory) / 'latest_primary.pth', *new_primary,
                                    device='cpu', companions={'correcting': new_corrector})
            self.assertEqual(state['companions_restored'], {'correcting'})
            for original, resumed in [(primary, new_primary), (corrector, new_corrector)]:
                self.assertEqual(original[2].state_dict(), resumed[2].state_dict())
                self.assertEqual(self.update(*original), self.update(*resumed))
                torch.testing.assert_close(original[0].weight, resumed[0].weight)


if __name__ == '__main__':
    unittest.main()
