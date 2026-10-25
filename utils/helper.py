import os 
import yaml
import pickle
import types

import torch
import torch.distributed as dist
from .lr_scheduler import Stage3PolynomialLR


class _UnusedAugmentationMetadata:
    """Consume legacy transform metadata without importing its old classes.

    Training builds its transforms from the current code. These checkpoint
    objects are never used as models, optimizers, schedulers, or augmentation.
    """

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass

    def __getattr__(self, name):
        # Albumentations also pickles bound methods in fields such as _key2func.
        # Pickle resolves getattr(transform, 'apply') before restoring its state.
        if name.startswith('__'):
            raise AttributeError(name)
        return _unused_augmentation_method


def _unused_augmentation_method(*args, **kwargs):
    return None


class _CheckpointUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'albumentations' or module.startswith('albumentations.'):
            return _UnusedAugmentationMetadata
        return super().find_class(module, name)


def _load_compat_pickle(file, **kwargs):
    return _CheckpointUnpickler(file, **kwargs).load()


# torch.load accepts a pickle module; keep compatibility local to this loader
# rather than modifying Albumentations or Python's global unpickling behavior.
_checkpoint_pickle = types.ModuleType('checkpoint_compat_pickle')
_checkpoint_pickle.Unpickler = _CheckpointUnpickler
_checkpoint_pickle.load = _load_compat_pickle


class Config:
    def __init__(self, config_dict):
        self.experiment = config_dict.get("experiment", {})
        self.data = config_dict.get("data", {})
        self.model = config_dict.get("model", {})
        self.training = config_dict.get("training", {})

    def __reper__(self):
        return f"Config(experiment={self.experiment}, data={self.data} model={self.model}, training={self.training})"
     

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    config = Config(config)
    return config

def save_checkpoint(epoch, model, optimizer, scheduler, cur_lr,
                    val_acc, config, train_transform, 
                    val_transform, save_dir, model_name="model", filename=None, companions=None):
    """Save training state; transform arguments remain for existing callers.

    Runtime augmentation objects are deliberately not serialized: their import
    paths change between Albumentations versions and they are rebuilt by train().
    """
    
    if dist.is_available() and dist.is_initialized():
        if dist.get_rank() != 0:
            return

    checkpoint_path = os.path.join(save_dir, filename or f'epoch{epoch}_{model_name}.pth')
    tmp_path = checkpoint_path + ".tmp"

    state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    torch.save({
            'epoch': epoch,
            'model_state_dict': state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'learning_rate': cur_lr,
            'val_accuracy': val_acc,
            'config': config,
            'companions': {
                name: {
                    'model_state_dict': extra_model.module.state_dict() if hasattr(extra_model, 'module') else extra_model.state_dict(),
                    'optimizer_state_dict': extra_optimizer.state_dict(),
                    'scheduler_state_dict': extra_scheduler.state_dict(),
                }
                for name, (extra_model, extra_optimizer, extra_scheduler) in (companions or {}).items()
            }

        }, tmp_path)
    os.replace(tmp_path, checkpoint_path)

    print(f"Epoch:{epoch} checkpoint has been saved at:{checkpoint_path}")

def load_checkpoint(checkpoint_path, model, optimizer, scheduler, device, resume=True, companions=None):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f'Checkpoint must point to a file: {checkpoint_path}')
    checkpoint = torch.load(
        checkpoint_path, 
        map_location=device,
        pickle_module=_checkpoint_pickle,
        weights_only=False
        )
    epoch = checkpoint['epoch']
    state_dict = checkpoint['model_state_dict']
    if hasattr(model, "module"):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)
    
    if resume:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler_state = checkpoint.get('scheduler_state_dict') or {}
        if isinstance(scheduler, Stage3PolynomialLR):
            scheduler.restore(scheduler_state, completed_epochs=epoch)
        elif scheduler is not None and scheduler_state:
            scheduler.load_state_dict(scheduler_state)

    restored = set()
    for name, (extra_model, extra_optimizer, extra_scheduler) in (companions or {}).items():
        saved = checkpoint.get('companions', {}).get(name)
        if saved is None:
            continue
        module = extra_model.module if hasattr(extra_model, 'module') else extra_model
        module.load_state_dict(saved['model_state_dict'])
        if resume:
            extra_optimizer.load_state_dict(saved['optimizer_state_dict'])
            if isinstance(extra_scheduler, Stage3PolynomialLR):
                extra_scheduler.restore(saved.get('scheduler_state_dict') or {}, epoch)
            else:
                extra_scheduler.load_state_dict(saved['scheduler_state_dict'])
        restored.add(name)
    
    state = {
        "epoch": epoch, 
        "model": model, 
        "optimizer": optimizer, 
        "scheduler": scheduler,
        "companions_restored": restored,
        "config": checkpoint.get('config'),
    }

    return state

def lr_vs_epoch(num_epochs, lrs, save_dir):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, num_epochs + 1), lrs, marker='o', linestyle='-')
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate vs. Epoch")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'epoch_vs_lr.png'), bbox_inches='tight')

def compute_class_distribution(masks, class_names, ignore_index=255):
    """
    Computes per-class pixel distribution.

    Args:
        masks:
            list of tensors with shape [B, H, W] or [H, W]

        class_names:
            list of class names

        ignore_index:
            ignored label value

    Returns:
        dict:
            {
                class_name: percentage
            }
    """

    import torch

    num_classes = len(class_names)

    class_counts = torch.zeros(num_classes, dtype=torch.long)

    for mask in masks:

        # Convert [H, W] -> [1, H, W]
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        valid_mask = mask != ignore_index
        valid_pixels = mask[valid_mask]

        counts = torch.bincount(
            valid_pixels.view(-1),
            minlength=num_classes
        )

        class_counts += counts.cpu()

    total_pixels = class_counts.sum().float()

    percentages = (
        class_counts.float() / (total_pixels + 1e-10)
    ) * 100

    distribution = {
        class_name: round(percentages[idx].item(), 4)
        for idx, class_name in enumerate(class_names)
    }

    return distribution
