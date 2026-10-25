import torch
import torch.distributed as dist

from utils.eval import compute_confusion_matrix, compute_iou_per_class
from utils.eval import compute_per_class_accuracy, plot_confusion_matrix, unsupervised_loss
from utils.helper import load_checkpoint


def initialize_stage3(paths, models, optimizers, schedulers, device, resume, mode):
    """Load stage-2 weights or resume a stage-3 primary/corrector bundle."""
    if mode == 'self_correction' and not all(paths.get(name) for name in ('primary', 'ancillary')):
        raise ValueError('Self-correction needs primary and ancillary checkpoint paths.')
    if resume and not paths.get('primary'):
        raise ValueError('Resuming stage 3 requires a primary checkpoint path.')
    state = None
    companions = ({'correcting': (models['correcting'], optimizers['correcting'], schedulers['correcting'])}
                  if mode == 'self_correction' else {})
    if paths.get('primary'):
        state = load_checkpoint(paths['primary'], models['primary'], optimizers['primary'],
                                schedulers['primary'], device, resume=resume, companions=companions)
        saved_config = state['config']
        saved_training = getattr(saved_config, 'training', {})
        previous_mode = saved_training.get('stage3_mode')
        if resume and previous_mode is not None and previous_mode != mode:
            raise ValueError('Stage 3 mode changed; use continue_traning=False to initialize a new experiment.')
    if mode == 'self_correction':
        load_checkpoint(paths['ancillary'], models['ancillary'], optimizers['ancillary'],
                        schedulers['ancillary'], device, resume=False)
        if state is None or 'correcting' not in state['companions_restored']:
            if not paths.get('correcting'):
                raise ValueError('This checkpoint has no corrector bundle; provide a correcting checkpoint path.')
            load_checkpoint(paths['correcting'], models['correcting'], optimizers['correcting'],
                            schedulers['correcting'], device, resume=False)
            if resume:
                # Legacy stage 3 froze the corrector. Start its new optimization
                # at the resumed epoch without importing stage-2 scheduler state.
                schedulers['correcting'].restore({}, state['epoch'])
    return state['epoch'] + 1 if resume else 1


def train_primary_model_epoch(epoch, data_loaders, device, models, optimizers,
                              loss_funcs, schedulers, accum_steps, logger,
                              mode='self_correction'):
    if mode not in ('self_correction', 'supervised'):
        raise ValueError(f'Unknown stage 3 mode: {mode}')
    if accum_steps < 1 or not len(data_loaders['w_loader']) or not len(data_loaders['f_loader']):
        raise ValueError('Stage 3 requires nonempty loaders and positive accumulation.')
    active = ['primary']
    models['primary'].train()
    if mode == 'self_correction':
        active.append('correcting')
        models['ancillary'].eval()
        for parameter in models['ancillary'].parameters():
            parameter.requires_grad_(False)
        for parameter in models['correcting'].parameters():
            parameter.requires_grad_(True)
    for name in active:
        optimizers[name].zero_grad()

    total_loss = 0.0
    f_iterator = iter(data_loaders['f_loader'])
    num_batches = len(data_loaders['w_loader'])
    for batch_idx, (w_imgs, w_targets) in enumerate(data_loaders['w_loader']):
        try:
            f_imgs, f_boxes, f_masks = next(f_iterator)
        except StopIteration:
            f_iterator = iter(data_loaders['f_loader'])
            f_imgs, f_boxes, f_masks = next(f_iterator)

        f_imgs, f_masks = f_imgs.to(device), f_masks.to(device).long()
        w_imgs, w_targets = w_imgs.to(device), w_targets.to(device)
        images = torch.cat([f_imgs, w_imgs], dim=0)
        logits = models['primary'](images)
        f_logits, w_logits = logits[:f_imgs.size(0)], logits[f_imgs.size(0):]
        supervised = loss_funcs['ce_loss'](f_logits, f_masks) + loss_funcs['dice_loss'](f_logits, f_masks)
        correction_loss = logits.new_zeros(())

        if mode == 'self_correction':
            if w_targets.ndim != 4 or w_targets.shape[1] != logits.shape[1]:
                raise ValueError('Self-correction requires bounding-box channels [B, C, H, W].')
            boxes = torch.cat([f_boxes.to(device).float(), w_targets.float()], dim=0)
            with torch.no_grad():
                ancillary = models['ancillary'](images, boxes)
                models['correcting'].eval()
                weak_targets = models['correcting'](w_logits.detach(), ancillary[f_imgs.size(0):])
            weak_loss = unsupervised_loss(w_logits, weak_targets)

            # The corrector learns from real labels; weak targets stay detached.
            models['correcting'].train()
            corrected_f = models['correcting'](f_logits.detach(), ancillary[:f_imgs.size(0)])
            correction_loss = (loss_funcs['ce_loss'](corrected_f, f_masks)
                               + loss_funcs['dice_loss'](corrected_f, f_masks))
        else:
            w_masks = w_targets.long()
            weak_loss = (loss_funcs['ce_loss'](w_logits, w_masks)
                         + loss_funcs['dice_loss'](w_logits, w_masks))

        loss = supervised + weak_loss + correction_loss
        total_loss += loss.item()
        group_start = (batch_idx // accum_steps) * accum_steps
        group_size = min(accum_steps, num_batches - group_start)
        (loss / group_size).backward()
        if (batch_idx + 1) % accum_steps == 0 or batch_idx + 1 == num_batches:
            for name in active:
                optimizers[name].step()
                optimizers[name].zero_grad()
                schedulers[name].step()

        if batch_idx % 200 == 0 or batch_idx == 20:
            logger.info(
                f'TRAIN STAGE 3: Epoch:{epoch} Batch:{batch_idx}/{num_batches} '
                f'Supervised:{supervised.item():.3f} Weak:{weak_loss.item():.3f} '
                f'Corrector:{correction_loss.item():.3f} Combined:{loss.item():.3f}'
            )
    average = total_loss / num_batches
    logger.info(f'PRIMARY MODEL Epoch:{epoch} average train loss:{average:.3f}')
    return average


def validate_primary_model(epoch, data_loader, device, models, loss_funcs,
                           class_names, logger, save_dir=None):
    models['primary'].eval()
    total_loss, samples, total_cm = 0.0, 0, None
    with torch.no_grad():
        for images, _, masks in data_loader:
            images, masks = images.to(device), masks.to(device).long()
            outputs = models['primary'](images)
            total_loss += loss_funcs['ce_loss'](outputs, masks).item() * images.size(0)
            samples += images.size(0)
            cm = compute_confusion_matrix(masks, outputs, class_names, ignore_index=255)
            total_cm = cm if total_cm is None else total_cm + cm
    if total_cm is None:
        raise ValueError('Validation loader is empty.')
    iou = compute_iou_per_class(total_cm)
    accuracy = compute_per_class_accuracy(total_cm)
    metrics = {
        'avg_loss': total_loss / samples,
        'iou_per_class': iou,
        'acc_per_class': accuracy,
        'primary_mIoU': iou.mean().item(),
        'primary_foreground_mIoU': iou[1:].mean().item(),
        'primary_avg_acc': accuracy.mean().item(),
        'primary_foreground_avg_acc': accuracy[1:].mean().item(),
    }
    logger.info(f'Epoch: {epoch} | Stage 3 validation')
    logger.info(metrics)
    if save_dir is not None and (not dist.is_initialized() or dist.get_rank() == 0):
        plot_confusion_matrix(total_cm, class_names, save_path=save_dir)
    return metrics
