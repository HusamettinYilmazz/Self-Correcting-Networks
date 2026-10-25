import math
import warnings

from torch.optim.lr_scheduler import LRScheduler


class Stage3PolynomialLR(LRScheduler):
    """Polynomial decay over optimizer updates, including across Kaggle sessions."""

    def __init__(self, optimizer, num_epochs, batches_per_epoch, accum_steps=1, power=0.9):
        if min(num_epochs, batches_per_epoch, accum_steps) <= 0:
            raise ValueError('Epochs, batches per epoch, and accumulation must be positive.')
        self.num_epochs = num_epochs
        self.updates_per_epoch = math.ceil(batches_per_epoch / accum_steps)
        self.total_steps = num_epochs * self.updates_per_epoch
        self.power = power
        self.start_step = 0
        self.start_lrs = [group['lr'] for group in optimizer.param_groups]
        self.schedule_version = 1
        super().__init__(optimizer)

    def get_lr(self):
        remaining = max(1, self.total_steps - self.start_step)
        progress = max(0, self.last_epoch - self.start_step) / remaining
        factor = max(0.0, 1.0 - progress) ** self.power
        return [lr * factor for lr in self.start_lrs]

    def restore(self, state_dict, completed_epochs):
        """Restore exactly, or migrate a legacy/changed schedule from its saved LR.

        Checkpoints are written at epoch boundaries. When loader length, GPU count,
        accumulation or epoch budget changes, use that completed epoch boundary in
        the new schedule and decay the saved LR over the remaining updates.
        """
        completed_steps = completed_epochs * self.updates_per_epoch
        compatible = (
            state_dict.get('schedule_version') == self.schedule_version
            and state_dict.get('num_epochs') == self.num_epochs
            and state_dict.get('updates_per_epoch') == self.updates_per_epoch
            and state_dict.get('last_epoch') == completed_steps
        )
        if compatible:
            self.load_state_dict(state_dict)
            return

        if completed_steps < self.total_steps and all(
            group['lr'] == 0 for group in self.optimizer.param_groups
        ):
            raise ValueError(
                'Checkpoint learning rate is zero. To fine-tune after a finished '
                'schedule, load weights with continue_traning=False and set a new learning rate.'
            )

        reason = 'legacy checkpoint' if 'schedule_version' not in state_dict else 'changed training budget or loader'
        warnings.warn(
            f'Stage 3 scheduler: {reason}; preserving the checkpoint learning rate '
            f'and decaying over the remaining epochs up to {self.num_epochs}.',
            stacklevel=2,
        )
        self.last_epoch = completed_steps
        self._step_count = completed_steps + 1
        self.start_step = completed_steps
        self.start_lrs = [group['lr'] for group in self.optimizer.param_groups]
        self._last_lr = list(self.start_lrs)
