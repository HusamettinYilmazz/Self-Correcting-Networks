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

def multiscale_flip_tta(model, imgs, bboxes, scales=(0.75, 1.0, 1.25), use_flip=True):
    """
    Returns:
        logits: [B, C, H, W]
    """

    # model.eval()

    # device = imgs.device
    # imgs = imgs.to(device)
    # bboxes = bboxes.to(device)

    B, C, H, W = imgs.shape
    num_scales = 0

    total_logits = None

    for s in scales:

        # -------------------------
        # scale image
        # -------------------------
        scaled_imgs = F.interpolate(
            imgs,
            scale_factor=s,
            mode="bilinear",
            align_corners=False,
            recompute_scale_factor=True
        )

        # -------------------------
        # scale bbox
        # -------------------------
        bboxes_scaled = bboxes.clone().float()
        bboxes_scaled[:, :, [0, 2]] *= s  # x1, x2
        bboxes_scaled[:, :, [1, 3]] *= s  # y1, y2

        logits = model(scaled_imgs, bboxes_scaled)

        # resize back to original resolution
        logits = F.interpolate(
            logits,
            size=(H, W),
            mode="bilinear",
            align_corners=False
        )

        total_logits = logits if total_logits is None else total_logits + logits
        num_scales += 1

        # -------------------------
        # FLIP TTA
        # -------------------------
        if use_flip:
            flipped_imgs = torch.flip(scaled_imgs, dims=[-1])

            W_s = scaled_imgs.shape[-1]

            bboxes_flipped = bboxes_scaled.clone()

            x1 = bboxes_scaled[:, :, 0].clone()
            x2 = bboxes_scaled[:, :, 2].clone()

            bboxes_flipped[:, :, 0] = W_s - x2
            bboxes_flipped[:, :, 2] = W_s - x1

            logits_f = model(flipped_imgs, bboxes_flipped)

            logits_f = torch.flip(logits_f, dims=[-1])

            logits_f = F.interpolate(
                logits_f,
                size=(H, W),
                mode="bilinear",
                align_corners=False
            )

            total_logits += logits_f
            num_scales += 1

    return total_logits / num_scales

def class_balanced_hard_mining(outputs, masks, k_ratio=0.2):
    loss_map = F.cross_entropy(
        outputs, masks,
        ignore_index=255,
        reduction='none'
    )  # (B, H, W)

    loss_flat = loss_map.view(-1)
    mask_flat = masks.view(-1)

    valid = mask_flat != 255

    loss_flat = loss_flat[valid]
    mask_flat = mask_flat[valid]

    k = int(k_ratio * loss_flat.numel())
    k = max(k, 1)

    hard_loss, _ = torch.topk(loss_flat, k)
    return hard_loss.mean()
