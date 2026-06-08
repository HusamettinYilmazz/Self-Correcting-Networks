from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn.functional as F

def plot_confusion_matrix(cm, class_names, save_path=None):
    cm_normalized = cm.numpy().astype(float) / (cm.numpy().sum(axis=1, keepdims=True) + 1e-10)
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(
        cm_normalized * 100,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        vmin=0,
        vmax=100
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title("Confusion Matrix (%)")

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Confusion matrix saved to {save_path}")
    plt.close(fig)

def compute_confusion_matrix(y_true, y_preds, class_names, ignore_index=255):
    """
    y_preds: [B, C, H, W] long
    y_true:  [B, H, W] long
    """
    y_preds = y_preds.argmax(dim=1)
    
    num_classes = len(class_names)

    y_preds = y_preds.cpu().numpy().flatten()
    y_true  = y_true.cpu().numpy().flatten()

    valid   = (y_true != ignore_index) & (y_true >= 0) & (y_true < num_classes)
    y_preds = y_preds[valid]
    y_true  = y_true[valid]

    cm = confusion_matrix(y_true, y_preds, labels=list(range(num_classes)))

    return torch.tensor(cm, dtype=torch.long)

def compute_iou_per_class(cm):
    return (cm.diag() / (cm.sum(dim=1) + cm.sum(dim=0) - cm.diag() + 1e-10)).cpu().numpy()

def compute_per_class_accuracy(cm):
    return (cm.diag() / (cm.sum(dim=1) + 1e-10)).cpu().numpy()

def get_boundaries(mask):
    """
    mask: [B,H,W] long
    returns: [B,H,W] bool
    """

    boundary = torch.zeros_like(mask, dtype=torch.bool)

    boundary[:, :, 1:] |= mask[:, :, 1:] != mask[:, :, :-1]
    boundary[:, :, :-1] |= mask[:, :, 1:] != mask[:, :, :-1]

    boundary[:, 1:, :] |= mask[:, 1:, :] != mask[:, :-1, :]
    boundary[:, :-1, :] |= mask[:, 1:, :] != mask[:, :-1, :]

    return boundary

def boundary_f1(pred, target):
    pred_b = get_boundaries(pred)
    target_b = get_boundaries(target)

    tp = (pred_b & target_b).sum().float()
    fp = (pred_b & ~target_b).sum().float()
    fn = (~pred_b & target_b).sum().float()

    precision = tp / (tp + fp + 1e-6)
    recall    = tp / (tp + fn + 1e-6)

    bf1 = 2 * precision * recall / (precision + recall + 1e-6)
    return bf1

def class_distribution(preds, num_classes):
    # preds: [B,H,W]
    hist = torch.zeros(num_classes, device=preds.device)

    for c in range(num_classes):
        hist[c] = (preds == c).sum()

    return hist / hist.sum()

def imbalance_indicator(preds, targets, num_classes):
    pred = class_distribution(preds, num_classes)
    gt   = class_distribution(targets, num_classes)

    return F.kl_div(
        (pred + 1e-8).log(),
        gt,
        reduction="batchmean"
    ).item()

def flip_image(x):
    return torch.flip(x, dims=[-1])


def flip_bboxes_xyxy(bboxes, img_width):
    """
    bboxes: [N, 4] in (x1, y1, x2, y2)
    """
    flipped = bboxes.clone()
    flipped[:, 0] = img_width - bboxes[:, 2]
    flipped[:, 2] = img_width - bboxes[:, 0]
    return flipped


def tta_flip_ancillary(model, imgs, bboxes=None):
    """
    Flip TTA for segmentation model with optional bbox input
    """
    logits1 = model(imgs, bboxes)

    imgs_flip = flip_image(imgs)

    if bboxes is not None:
        bboxes_flip = flip_bboxes_xyxy(bboxes, imgs.shape[-1])
    else:
        bboxes_flip = None

    logits2 = model(imgs_flip, bboxes_flip)
    logits2 = flip_image(logits2)

    return (logits1 + logits2) / 2
