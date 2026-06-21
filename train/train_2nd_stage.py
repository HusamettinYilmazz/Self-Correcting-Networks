import torch

from utils.eval import compute_confusion_matrix, compute_iou_per_class
from utils.eval import compute_per_class_accuracy, plot_confusion_matrix
from utils.eval import ce_weight

def train_correction_model_epoch(epoch, data_loader, device, models, optimizers, 
                                 loss_funcs, schedulers, accum_steps, logger):
    
    total_primary_loss, total_correcting_loss = 0.0, 0.0
    
    optimizers["primary"].zero_grad()
    optimizers["correcting"].zero_grad()

    models["primary"].train()
    models["correcting"].train()
    ancillary_module = models["ancillary"].module if hasattr(models["ancillary"], "module") else models["ancillary"]
    ancillary_module.freeze()
    models["ancillary"].eval()
    for batch_idx, (imgs, bboxs, masks) in enumerate(data_loader):
        imgs = imgs.to(device)
        bboxs = bboxs.to(device)
        masks = masks.to(device).long()
        
        primary_logits = models["primary"](imgs)
        with torch.no_grad():
            ancillary_logits = models["ancillary"](imgs, bboxs)

        correcting_logits = models["correcting"](
            primary_logits.detach(), 
            ancillary_logits.detach()
        )

        primary_ce_loss = loss_funcs["ce_loss"](primary_logits, masks)
        correcting_ce_loss = loss_funcs["ce_loss"](correcting_logits, masks)

        primary_dice_loss = loss_funcs["dice_loss"](primary_logits, masks)
        correcting_dice_loss = loss_funcs["dice_loss"](correcting_logits, masks)

        correcting_dice_weight = ce_weight(epoch=epoch, max_epoch=300) ** -1
        primary_loss = primary_ce_loss + primary_dice_loss
        correcting_loss = correcting_ce_loss + correcting_dice_weight * correcting_dice_loss

        total_primary_loss += primary_loss.item()
        total_correcting_loss += correcting_loss.item()

        acc_primary_loss = primary_loss / accum_steps
        acc_correcting_loss = correcting_loss / accum_steps

        acc_primary_loss.backward()
        acc_correcting_loss.backward()

        if (batch_idx+1) % accum_steps == 0 or (batch_idx+1) == len(data_loader):
            optimizers["primary"].step()
            optimizers["primary"].zero_grad()
            schedulers["primary"].step()
            
            optimizers["correcting"].step()
            optimizers["correcting"].zero_grad()
            schedulers["correcting"].step()
            

        if batch_idx % 20 == 0:
            logger.info(f"TRAIN: Epoch:{epoch} at Batch:{batch_idx}/{len(data_loader)} Primary model loss:{primary_ce_loss.item():.3f} | Primary Dice Loss:{primary_dice_loss.item():.3f} | Primary Loss:{primary_loss.item():.3f}")
            logger.info(f"Correcting network loss:{correcting_ce_loss.item():.3f} | Correcting Dice Loss:{correcting_dice_loss.item():.3f} | Correcting Loss:{correcting_loss.item():.3f}")
    primary_avg_loss = total_primary_loss/ len(data_loader)
    correcting_avg_loss = total_correcting_loss/ len(data_loader)
    logger.info(f"PRIMARY MODEL Epoch:{epoch} average train Loss:{primary_avg_loss:.3f}")
    logger.info(f"CORRECTING NETWORK Epoch:{epoch} average train Loss:{correcting_avg_loss:.3f}")
    logger.info(f"Epoch:{epoch} average train Loss:{(primary_avg_loss+correcting_avg_loss)/2:.3f}")
    
    return primary_avg_loss, correcting_avg_loss

def validate_correction_model(epoch, data_loader, device, models, loss_funcs, class_names, logger, save_dir=None):
    total_primary_loss = 0.0
    total_correcting_loss = 0.0
    primary_total_cm, correcting_total_cm = None, None

    models["primary"].eval()
    models["correcting"].eval()
    models["ancillary"].eval()
    with torch.no_grad():
        for imgs, bboxes, masks in data_loader:
            imgs = imgs.to(device)
            bboxes = bboxes.to(device)
            masks = masks.to(device).long()

            primary_logits = models["primary"](imgs)
            ancillary_logits = models["ancillary"](imgs, bboxes)
            correcting_logits = models["correcting"](primary_logits, ancillary_logits)
            
            primary_loss = loss_funcs["ce_loss"](primary_logits, masks)
            correcting_loss = loss_funcs["ce_loss"](correcting_logits, masks)

            total_primary_loss += primary_loss.item()
            total_correcting_loss += correcting_loss.item()

            # primary_preds = primary_logits.argmax(dim=1)
            # correcting_preds = correcting_logits.argmax(dim=1)
            
            ## it has no real meaning (the training isn't done for this)
            primary_cm = compute_confusion_matrix(
                masks,
                primary_logits,
                class_names,
                ignore_index=255
            )

            correcting_cm = compute_confusion_matrix(
                masks,
                correcting_logits,
                class_names,
                ignore_index=255
            )

            primary_total_cm = primary_cm if primary_total_cm is None \
                                    else primary_total_cm + primary_cm
            
            correcting_total_cm = correcting_cm if correcting_total_cm is None \
                                    else correcting_total_cm + correcting_cm

    primary_iou_per_class = compute_iou_per_class(primary_total_cm)
    primary_acc_per_class = compute_per_class_accuracy(primary_total_cm)

    correcting_iou_per_class = compute_iou_per_class(correcting_total_cm)
    correcting_acc_per_class = compute_per_class_accuracy(correcting_total_cm)

    metrics = {
        "primary_avg_loss": total_primary_loss / len(data_loader),
        "correcting_avg_loss": total_correcting_loss / len(data_loader),

        "primary_acc_per_class": primary_acc_per_class,
        "primary_iou_per_class": primary_iou_per_class,
        
        "correcting_acc_per_class": correcting_acc_per_class,
        "correcting_iou_per_class": correcting_iou_per_class,

        "primary_mIoU":     primary_iou_per_class[1:].mean().item(),
        "primary_avg_acc":  primary_acc_per_class[1:].mean().item(),

        "correcting_mIoU":     correcting_iou_per_class[1:].mean().item(),
        "correcting_avg_acc":  correcting_acc_per_class[1:].mean().item(),
    }

    if save_dir is not None:
        plot_confusion_matrix(correcting_total_cm, class_names, save_path=save_dir)

    logger.info(f"Epoch: {epoch} | Stage 2 validation")
    logger.info(metrics)

    return metrics
