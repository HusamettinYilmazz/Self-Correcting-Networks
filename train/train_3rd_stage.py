import torch

from utils.eval import compute_confusion_matrix, compute_iou_per_class
from utils.eval import compute_per_class_accuracy, plot_confusion_matrix
from utils.eval import unsupervised_loss, soft_dice_unsup

def compute_lambda(epoch):  
    """
    needs to be revised for both its importance and 
    the number of epochs = 10 needs to passed as 3rd stage num_epoch.
    """
    return min(1.0, epoch/10)

def train_primary_model_epoch(epoch, data_loaders, device, models, optimizers,
                              loss_funcs, schedulers, accum_steps, logger):
    
    total_loss = 0.0
    
    optimizers["primary"].zero_grad()
    models["primary"].train()
    correcting_module = models["correcting"].module if hasattr(models["correcting"], "module") else models["correcting"]
    correcting_module.freeze()
    models["correcting"].eval()
    ancillary_module = models["ancillary"].module if hasattr(models["ancillary"], "module") else models["ancillary"]
    ancillary_module.freeze()
    models["ancillary"].eval()
    
    f_loader_iter = iter(data_loaders["f_loader"])
    epochs_f_loder = 1
    for batch_idx, (w_imgs, w_masks) in enumerate(data_loaders["w_loader"]):
        try:
            f_imgs, _, f_masks = next(f_loader_iter)
        except StopIteration:
            epochs_f_loder += 1
            logger.info(f"f_loader epoch:{epochs_f_loder} starting")
            # Restart the iterator cleanly when it runs out of data
            f_loader_iter = iter(data_loaders["f_loader"])
            f_imgs, _, f_masks = next(f_loader_iter)

        f_imgs, f_masks = f_imgs.to(device), f_masks.to(device).long()
        w_imgs, w_masks = w_imgs.to(device), w_masks.to(device)

        f_primary_logits = models["primary"](f_imgs)
        w_primary_logits = models["primary"](w_imgs)

        with torch.no_grad():
            ancillary_outputs = models["ancillary"](w_imgs, w_masks)
            correcting_logits = models["correcting"](
                w_primary_logits.detach(), 
                ancillary_outputs.detach()
            )
            correcting_logits = correcting_logits.detach()

        primary_ce_loss = loss_funcs["ce_loss"](f_primary_logits, f_masks)
        primary_dice_loss = loss_funcs["dice_loss"](f_primary_logits, f_masks)
        primary_loss = primary_ce_loss + primary_dice_loss
        unsup_ce_loss = unsupervised_loss(w_primary_logits, correcting_logits)
        # unsup_dice_loss = soft_dice_unsup(w_primary_logits, correcting_logits)
        unsup_loss = unsup_ce_loss # + unsup_dice_loss
        # lambda_u = compute_lambda(epoch)

        loss = primary_loss + unsup_loss
        
        total_loss += loss.item()

        acc_loss = loss / accum_steps
        acc_loss.backward()

        if (batch_idx+1) % accum_steps == 0 or (batch_idx+1) == len(data_loaders["w_loader"]):
            optimizers["primary"].step()
            optimizers["primary"].zero_grad()
            schedulers["primary"].step()


        if batch_idx % 200 == 0 or batch_idx == 20:
            logger.info(f"TRAIN PRIMARY MODEL: Epoch:{epoch} at Batch:{batch_idx}/{len(data_loaders['w_loader'])} Primary Loss:{primary_loss.item():.3f} | Unsupervised Loss:{unsup_loss.item():.3f} & Combined Loss:{loss.item():.3f}")
    
    logger.info(f"fully supervised dataset trained: {epochs_f_loder} epochs")
    avg_loss = total_loss/ len(data_loaders["w_loader"])
    logger.info(f"PRIMARY MODEL Epoch:{epoch} average train Loss:{avg_loss:.3f}")
    
    return avg_loss

def validate_primary_model(epoch, data_loader, device, models, loss_funcs,
                     class_names, logger, save_dir=None):
    
    models["primary"].eval()

    total_loss = 0.0
    total_cm = None

    with torch.no_grad():
        for imgs, _, masks in data_loader:
            imgs = imgs.to(device)
            masks = masks.to(device).long()


            outputs = models["primary"](imgs)

            loss = loss_funcs['ce_loss'](outputs, masks)
            total_loss += loss.item()

            # preds = outputs.argmax(dim=1)

            cm = compute_confusion_matrix(
                masks,
                outputs,
                class_names,
                ignore_index=255
            )

            total_cm = cm if total_cm is None else total_cm + cm

    iou_per_class = compute_iou_per_class(total_cm)
    acc_per_class = compute_per_class_accuracy(total_cm)

    metrics = {
        "avg_loss": total_loss / len(data_loader),
        "iou_per_class": iou_per_class,
        "acc_per_class": acc_per_class,
        "primary_mIoU"         : iou_per_class[1:].mean().item(),
        "primary_avg_acc"      : acc_per_class[1:].mean().item(),
    }

    logger.info(f"Epoch: {epoch} | Stage 3 validation")
    logger.info(metrics)

    if save_dir is not None:
        plot_confusion_matrix(total_cm, class_names, save_path=save_dir)

    return metrics
