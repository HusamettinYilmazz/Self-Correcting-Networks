import os 
import yaml
import pickle

import matplotlib.pyplot as plt
import torch
import torch.distributed as dist

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
                    val_transform, save_dir, model_name="model"):
    
    if dist.is_available() and dist.is_initialized():
        if dist.get_rank() != 0:
            return

    checkpoint_path = os.path.join(save_dir, f'epoch{epoch}_{model_name}.pth')
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
            'train_transform': train_transform,
            'val_transform': val_transform

        }, tmp_path)
    os.replace(tmp_path, checkpoint_path)

    print(f"Epoch:{epoch} checkpoint has been saved at:{checkpoint_path}")

def load_checkpoint(checkpoint_path, model, optimizer, scheduler, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    epoch = checkpoint['epoch']
    state_dict = checkpoint['model_state_dict']
    if hasattr(model, "module"):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if ('scheduler_state_dict' in checkpoint and 
        checkpoint['scheduler_state_dict'] is not None):

        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])


    for param_group in optimizer.param_groups:
        param_group['lr'] = checkpoint['learning_rate']
    
    state = {
        "epoch": epoch, 
        "model": model, 
        "optimizer": optimizer, 
        "scheduler": scheduler
    }

    return state

def lr_vs_epoch(num_epochs, lrs, save_dir):
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
