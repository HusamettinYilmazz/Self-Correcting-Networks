import torch
from utils.eval import compute_confusion_matrix, plot_confusion_matrix
from utils.eval import compute_per_class_accuracy,  compute_iou_per_class
from utils.eval import boundary_f1, imbalance_indicator
from utils.eval import multiscale_flip_tta, class_balanced_hard_mining

def train_ancillary_model_epoch(epoch, data_loader, device, models, optimizers, loss_funcs, schedulers, accum_steps, logger):
    total_loss = 0.0
    preds, targets = [], []

    optimizers["ancillary"].zero_grad()
    models["ancillary"].train()
    for batch_idx, (imgs, bboxs, masks) in enumerate(data_loader):
        imgs, bboxs, masks = imgs.to(device), bboxs.to(device), masks.to(device).long()
        
        outputs = models["ancillary"](imgs, bboxs)

        if epoch > 100:
            ce_loss = class_balanced_hard_mining(
                            outputs=outputs,
                            masks= masks,
                            k_ratio=0.3
                        )
        else:
            ce_loss = loss_funcs["ce_loss"](outputs, masks)
        dice_loss = loss_funcs["dice_loss"](outputs, masks)
        loss = ce_loss + 0.5 * dice_loss
        total_loss += loss.item()

        acc_loss = loss / accum_steps
        acc_loss.backward()
        

        if (batch_idx+1) % accum_steps == 0 or (batch_idx+1) == len(data_loader):
            optimizers["ancillary"].step()
            optimizers["ancillary"].zero_grad()
            schedulers["ancillary"].step()
        

        pred_classes = torch.argmax(outputs, dim=1)
        preds.extend(pred_classes.detach().cpu())
        targets.extend(masks.detach().cpu())
        if batch_idx % 20 == 0:
            b_f1 = boundary_f1(
                pred_classes.detach().cpu(),
                masks.detach().cpu()
            )
            logger.info(f"TRAIN: Epoch:{epoch} at Batch:{batch_idx}/{len(data_loader)} CrossEntropy Loss:{ce_loss.item():.3f} | Dice Loss:{dice_loss.item():.3f} | Loss:{loss.item():.3f} | Boundry F1: {b_f1.item():.3f}")
    
    avg_loss = total_loss/ len(data_loader)
    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)
    imbalance_ind = imbalance_indicator(preds, targets, 21)
    logger.info(f"Imblance Indicator: {imbalance_ind}")
    logger.info(f"ANCILLARY MODEL Epoch:{epoch} average train Loss:{avg_loss:.3f}")
    
    return avg_loss

def validate_ancillary_model(epoch, data_loader, device, models, loss_funcs, class_names, logger, save_dir=None):
    total_loss = 0.0
    total_ce_loss = 0.0
    total_dice_loss = 0.0
    total_cm = None
    preds_list, targets_list = [], []

    models["ancillary"].eval()
    with torch.no_grad():
        for imgs, bboxes, masks in data_loader:
            imgs = imgs.to(device)
            bboxes = bboxes.to(device)
            masks = masks.to(device).long()

            outputs = multiscale_flip_tta(
                model=models["ancillary"], 
                imgs=imgs, 
                bboxes=bboxes, 
                scales=(1, ),
                use_flip=False
                )

            ce_loss = loss_funcs["ce_loss"](outputs, masks)
            total_ce_loss += ce_loss.item()
            dice_loss = loss_funcs["dice_loss"](outputs, masks)
            total_dice_loss += dice_loss.item()
            loss = ce_loss + dice_loss
            total_loss += loss.item()

            cm = compute_confusion_matrix(
                masks,
                outputs,
                class_names,
                ignore_index=255
            )

            total_cm = cm if total_cm is None else total_cm + cm
            # predictions
            pred_classes = torch.argmax(outputs, dim=1)

            preds_list.append(pred_classes.cpu())
            targets_list.append(masks.cpu())

    iou = compute_iou_per_class(total_cm)
    acc = compute_per_class_accuracy(total_cm)

    preds = torch.cat(preds_list, dim=0)
    targets = torch.cat(targets_list, dim=0)
    b_f1 = boundary_f1(preds, targets)
    imbalance_ind = imbalance_indicator(preds, targets, 21)  

    metrics = {
        "avg_ce_loss": total_ce_loss / len(data_loader),
        "avg_dice_loss": total_dice_loss / len(data_loader),
        "avg_loss": total_loss / len(data_loader),
        "acc_per_class": acc,
        "iou_per_class": iou,
        "mIoU":     iou[1:].mean().item(),
        "avg_acc":  acc[1:].mean().item(),
        "Boundry F1": b_f1.item(),
        "Class Imbalancing Indicator": imbalance_ind,
    }

    if save_dir is not None:
        plot_confusion_matrix(total_cm[1:, 1:], class_names[1:], save_path=save_dir)

    logger.info(f"Epoch: {epoch} | Stage 1 validation")
    logger.info(metrics)

    return metrics
