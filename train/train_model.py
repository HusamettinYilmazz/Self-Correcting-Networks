import os
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import SGD, AdamW, lr_scheduler
from torch.cuda.amp import autocast, GradScaler

import albumentations as A
from albumentations.pytorch import ToTensorV2


from datasets import VOCDataset, SBDDataset
from utils import Config, load_config, lr_vs_epoch, save_checkpoint, load_checkpoint
from utils.logger import Logger
from utils.dice_loss import DiceLoss

from models.primary_model import PrimarySegmentationModel
from models.ancillary_model import AncillarySegmentationModel
from models.self_correcting_model import SelfCorrectingNetwrokFactory

from train_1st_stage import train_ancillary_model_epoch, validate_ancillary_model
from train_2nd_stage import train_correction_model_epoch, validate_correction_model
from train_3rd_stage import train_primary_model_epoch, validate_primary_model


def stage1_training_loop(starting_epoch, config: Config, train_loaders, val_loader, 
                         train_transform, val_transform, device, models,
                         optimizers, schedulers, loss_funcs, scaler, logger, save_dir):
    
    lrs = []
    best_miou = 0.0
    logger.info("Stage 1: Ancillary Model Training")
    logger.info(f"Stage 1 training dataset size: {len(train_loaders['f1_loader'])*config.training['batch_size']}")
    for epoch in range(starting_epoch, config.training['stage1_num_epochs']+1):
        logger.info(f"Epoch: {epoch}/{config.training['stage1_num_epochs']}")
        
        _ = train_ancillary_model_epoch(
                    epoch=epoch,
                    data_loader=train_loaders['f1_loader'], 
                    device=device,
                    models=models,
                    optimizers=optimizers,
                    loss_funcs=loss_funcs,
                    schedulers=schedulers,
                    accum_steps= config.training['grad_acc_steps'],
                    logger=logger
                )

        save_file = os.path.join(save_dir, f'epoch{epoch}_conf_matrix.png')
        val_metrics = validate_ancillary_model(
                        epoch=epoch,
                        data_loader=val_loader,
                        device=device,
                        models=models,
                        loss_funcs=loss_funcs,
                        class_names= config.model["class_labels"],
                        logger=logger,
                        save_dir=save_file
                    )

        
        
        logger.info(f"Current learning rate: {optimizers['ancillary'].param_groups[0]['lr']}")
        # schedulers["ancillary"].step(val_metrics['avg_loss'])
        
        cur_lr = optimizers['ancillary'].param_groups[0]['lr']
        lrs.append(cur_lr)
        
        if val_metrics['mIoU'] > 0.5 and (epoch % 30 == 0 or round(val_metrics['mIoU'], 2) > round(best_miou, 2)):
            save_checkpoint(
                epoch= epoch, 
                model = models["ancillary"],
                optimizer = optimizers['ancillary'], 
                scheduler = schedulers['ancillary'],
                cur_lr = cur_lr, 
                val_acc = val_metrics['acc_per_class'], 
                config = config, 
                train_transform = train_transform, 
                val_transform = val_transform, 
                save_dir = save_dir,
                model_name= "ancillary"
            )
            if epoch % 30 == 0:
                ...
            else:
                best_miou = val_metrics['mIoU']
        
    logger.info(f"First stage training completed successfully")

    lr_vs_epoch(config.training['stage1_num_epochs']-starting_epoch+1, lrs, save_dir)

    return 

def stage2_training_loop(starting_epoch, config: Config, train_loaders, val_loader, 
                         train_transform, val_transform, device, models,
                         optimizers, schedulers, loss_funcs, scaler, logger, save_dir):
    
    prim_lrs, corr_lrs = [], []
    best_miou = 0.0
    logger.info("Stage 2: Primary Model and Self Correcting Network Training")
    logger.info(f"Stage 2 training dataset size: {len(train_loaders['f_loader'])*config.training['batch_size']}")
    for epoch in range(starting_epoch, config.training['stage2_num_epochs']+1):
        logger.info(f"Epoch: {epoch}/{config.training['stage2_num_epochs']}")
        
        _, _ = train_correction_model_epoch(
                    epoch=epoch,
                    data_loader=train_loaders['f_loader'],
                    device=device,
                    models=models,
                    optimizers=optimizers,
                    loss_funcs=loss_funcs,
                    schedulers=schedulers,
                    accum_steps=config.training['grad_acc_steps'],
                    logger=logger
                )

        save_file = os.path.join(save_dir, f'epoch{epoch}_conf_matrix.png')
        val_metrics = validate_correction_model(
                        epoch=epoch,
                        data_loader=val_loader,
                        device=device,
                        models=models,
                        loss_funcs=loss_funcs,
                        class_names=config.model["class_labels"],
                        logger=logger,
                        save_dir=save_file
                    )

        
        
        logger.info(f"Current learning rate for primary model: {optimizers['primary'].param_groups[0]['lr']}")
        logger.info(f"Current learning rate for correcting network: {optimizers['correcting'].param_groups[0]['lr']}")
        
        # schedulers["primary"].step(val_metrics['primary_avg_loss'])
        # schedulers["correcting"].step(val_metrics['correcting_avg_loss'])
        
        prim_lr = optimizers['primary'].param_groups[0]['lr']
        corr_lr = optimizers['correcting'].param_groups[0]['lr']

        prim_lrs.append(prim_lr)
        corr_lrs.append(corr_lr)
        
        if val_metrics['correcting_mIoU'] > 0.6 and (epoch % 30 == 0 or round(val_metrics['correcting_mIoU'], 2) > round(best_miou, 2)):
            save_checkpoint(
                epoch= epoch, 
                model= models["primary"],
                optimizer= optimizers['primary'], 
                scheduler= schedulers['primary'],
                cur_lr= prim_lr, 
                val_acc= val_metrics['primary_acc_per_class'], 
                config= config, 
                train_transform= train_transform, 
                val_transform= val_transform, 
                save_dir= save_dir,
                model_name= "primary"
            )

            save_checkpoint(
                epoch= epoch, 
                model= models["correcting"],
                optimizer= optimizers['correcting'], 
                scheduler= schedulers['correcting'],
                cur_lr= prim_lr, 
                val_acc= val_metrics['correcting_acc_per_class'], 
                config= config, 
                train_transform= train_transform, 
                val_transform= val_transform, 
                save_dir= save_dir,
                model_name= "correcting"
            )
            if epoch % 30 == 0:
                ...
            else:
                best_miou = val_metrics['correcting_mIoU']
        
    logger.info(f"Second stage training completed successfully")

    lr_vs_epoch(config.training['stage2_num_epochs']-starting_epoch+1, prim_lrs, save_dir)
    lr_vs_epoch(config.training['stage2_num_epochs']-starting_epoch+1, corr_lrs, save_dir)

def stage3_training_loop(starting_epoch, config: Config, train_loaders, val_loader, 
                         train_transform, val_transform,device, models,
                         optimizers, schedulers, loss_funcs, scaler, logger, save_dir):
    
    lr = []
    best_miou = 0.0
    logger.info("Stage 3: Primary Model Training")
    logger.info(f"Stage 3 training dataset size: {(len(train_loaders['f_loader'])+len(train_loaders['w_loader']))*config.training['batch_size']}")
    for epoch in range(starting_epoch, config.training['stage3_num_epochs']+1):
        logger.info(f"Epoch: {epoch}/{config.training['stage3_num_epochs']}")
        _ = train_primary_model_epoch(
                epoch=epoch,
                data_loaders=train_loaders,
                device=device,
                models=models,
                optimizers=optimizers,
                loss_funcs=loss_funcs,
                schedulers=schedulers,
                accum_steps=config.training['grad_acc_steps'],
                logger=logger
            )

        save_file = os.path.join(save_dir, f'epoch{epoch}_conf_matrix.png')
        val_metrics = validate_primary_model(
                        epoch=epoch,
                        data_loader=val_loader,
                        device=device,
                        models=models,
                        loss_funcs=loss_funcs,
                        class_names= config.model["class_labels"],
                        logger=logger,
                        save_dir=save_file
                    )

        
        
        logger.info(f"Current learning rate: {optimizers['primary'].param_groups[0]['lr']}")
        # schedulers['primary'].step(val_metrics['avg_loss'])
        
        cur_lr = optimizers['primary'].param_groups[0]['lr']
        lr.append(cur_lr)
        
        if val_metrics['primary_mIoU'] > 0.6 and (epoch % 30 == 0 or round(val_metrics['primary_mIoU'], 2) > round(best_miou, 2)):
            save_checkpoint(
                epoch= epoch, 
                model= models["primary"],
                optimizer= optimizers['primary'], 
                scheduler= schedulers['primary'],
                cur_lr= cur_lr, 
                val_acc= val_metrics['acc_per_class'], 
                config= config, 
                train_transform= train_transform, 
                val_transform= val_transform, 
                save_dir= save_dir,
                model_name= "primary"
            )
            if epoch % 30 == 0:
                ...
            else:
                best_miou = val_metrics['primary_mIoU']
        
    logger.info(f"Third stage training completed successfully")

    lr_vs_epoch(config.training['stage3_num_epochs']-starting_epoch+1, lr, save_dir)


def train(config: Config, checkpoint_path=None):

    dataset_path = os.path.join(ROOT, config.data['dataset_path'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_transform = A.Compose([
        A.RandomScale(scale_limit=0.5, p=1.0),

        A.PadIfNeeded(min_height=520, min_width=520),

        A.RandomCrop(512, 512),

        A.HorizontalFlip(p=0.5),

        A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1,
            p=0.5
        ),

        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.Resize(512, 512),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])
    
    train_val_dataset_path = os.path.join(dataset_path, 
                                        config.data['train_dataset_path'])
    
    weak_dataset_path = os.path.join(dataset_path, 
                                        config.data['weak_dataset_path'])

    fully_sup_train_dataset = VOCDataset(data_path= train_val_dataset_path,
                                    data_type="train",
                                    is_sup= True,
                                    split_ratio=1.0,
                                    transform=train_transform)
    generator = torch.Generator().manual_seed(42)
    f1_dataset, f2_dataset = random_split(
        fully_sup_train_dataset,
        [
            len(fully_sup_train_dataset) // 2,
            len(fully_sup_train_dataset) - len(fully_sup_train_dataset) // 2
        ],
        generator=generator
    )

    weak_train_dataset = SBDDataset(
        data_path= weak_dataset_path,
        transform=train_transform
    )
    
    val_dataset = VOCDataset(data_path= train_val_dataset_path,
                               data_type="val",
                               is_sup= True,
                               transform=val_transform)
    
    train_loaders = {
        "f_loader": DataLoader(
            dataset=fully_sup_train_dataset, 
            batch_size=config.training['batch_size'],
            shuffle= True, pin_memory= True,
            num_workers=4
            ),

        "w_loader": DataLoader(
            dataset=weak_train_dataset, 
            batch_size=config.training['batch_size'],
            shuffle= True, pin_memory= True,
            num_workers=4
            ),

        "f1_loader": DataLoader(
            dataset=f1_dataset, 
            batch_size=config.training['batch_size'],
            shuffle= True, pin_memory= True,
            num_workers=4
            ),

        "f2_loader": DataLoader(
            dataset=f2_dataset, 
            batch_size=config.training['batch_size'],
            shuffle= True, pin_memory= True,
            num_workers=4
            )
    }
    val_loader = DataLoader(dataset=val_dataset, 
                            batch_size=config.training['batch_size'], 
                            shuffle= True, pin_memory= True,
                            num_workers=4)

    models = {
        "primary": PrimarySegmentationModel(
            num_classes=config.model["num_classes"]
        ),
        
        "ancillary": AncillarySegmentationModel(
            num_classes=config.model["num_classes"]
        ),
        
        "correcting": SelfCorrectingNetwrokFactory().build_correction_module(
            variant = config.model['correcting_variant'], 
            num_classes=config.model["num_classes"]
        )
    }

    if config.training['training_stage'] == 1:
        if torch.cuda.device_count() > 1:
            models['ancillary'] = torch.nn.DataParallel(models['ancillary'])
        models['ancillary'] = models['ancillary'].to(device)

    else:
        if torch.cuda.device_count() > 1:
            if config.training['training_stage'] == 2:        
                models['primary'] = torch.nn.DataParallel(models['primary'])
                models['correcting'] = torch.nn.DataParallel(models['correcting'])
            
            elif config.training['training_stage'] == 3:
                models['primary'] = torch.nn.DataParallel(models['primary'])
            
            else: ## if all stages trained in one time
                models['ancillary'] = torch.nn.DataParallel(models['ancillary'])
                models['primary'] = torch.nn.DataParallel(models['primary'])
                models['correcting'] = torch.nn.DataParallel(models['correcting'])
            

        models['primary'] = models['primary'].to(device)
        models['ancillary'] = models['ancillary'].to(device)
        models['correcting'] = models['correcting'].to(device)
    

    optimizers = {
        "primary": AdamW(models['primary'].parameters(), 
                         lr=float(config.training['learning_rate']),
                         weight_decay=float(config.training['weight_decay'])
        ),

        "ancillary": AdamW(models['ancillary'].parameters(), 
                           lr=float(config.training['learning_rate']),
                           weight_decay=float(config.training['weight_decay'])
        ),

        "correcting": AdamW(models['correcting'].parameters(), 
                            lr=float(config.training['correcting_lr']),
                            weight_decay=float(config.training['weight_decay'])
        ),
    }
    
    if config.training['training_stage'] == 2:
        primary_scheduler = lr_scheduler.LambdaLR(
            optimizers['primary'],
            lr_lambda=lambda it: (1 - 
                                  it / (config.training['stage2_num_epochs'] * 
                                        len(train_loaders['f_loader']) /
                                        config.training['grad_acc_steps'])
                                  ) ** 0.9
        )
    else:
        primary_scheduler = lr_scheduler.LambdaLR(
            optimizers['primary'],
            lr_lambda=lambda it: (1 - 
                                  it / (config.training['stage3_num_epochs'] * 
                                        (len(train_loaders['f_loader']) + 
                                        len(train_loaders['w_loader'])) / 
                                        config.training['grad_acc_steps'])
                                  ) ** 0.9
        )

    schedulers = {
        "primary": primary_scheduler,

        "ancillary": lr_scheduler.LambdaLR(
            optimizers['ancillary'],
            lr_lambda=lambda it: (1 - 
                                  it / (config.training['stage1_num_epochs'] * 
                                        len(train_loaders['f1_loader']) / 
                                        config.training['grad_acc_steps'])
                                  ) ** 0.9
        ),

        "correcting": lr_scheduler.LambdaLR(
            optimizers['correcting'],
            lr_lambda=lambda it: (1 - 
                                  it / (config.training['stage2_num_epochs'] * 
                                        len(train_loaders['f_loader']) / 
                                        config.training['grad_acc_steps'])
                                  ) ** 0.9
        ),
    }

    weights = torch.tensor([
        0.2 if i == 0 else 1.0
        for i in range(config.model["num_classes"])
    ], dtype=torch.float32).to(device)

    loss_funcs = {
        "ce_loss": nn.CrossEntropyLoss(ignore_index=255, weight=weights),
        "dice_loss": DiceLoss(ignore_index=255, ignore_background=True)
    }
    scaler = GradScaler()

    if checkpoint_path:
        if config.training['training_stage'] == 1:
            if config.training['continue_traning']:
                ancillary_state = load_checkpoint(
                    checkpoint_path= checkpoint_path['ancillary'], 
                    model= models['ancillary'],
                    optimizer= optimizers['ancillary'], 
                    scheduler= schedulers['ancillary'], 
                    device= device
                )

                starting_epoch = ancillary_state['epoch'] + 1
                models['ancillary'] = ancillary_state['model']
                optimizers['ancillary'] = ancillary_state['optimizer']
                schedulers['ancillary'] = ancillary_state['scheduler']
            else:
                starting_epoch = 1

        elif config.training['training_stage'] == 2:
            if config.training['continue_traning']:
                primary_state = load_checkpoint(
                    checkpoint_path= checkpoint_path['primary'], 
                    model= models['primary'],
                    optimizer= optimizers['primary'], 
                    scheduler= schedulers['primary'], 
                    device= device
                )
                models['primary'] = primary_state['model']
                optimizers['primary'] = primary_state['optimizer']
                schedulers['primary'] = primary_state['scheduler']

                correcting_state = load_checkpoint(
                    checkpoint_path= checkpoint_path['correcting'], 
                    model= models['correcting'],
                    optimizer= optimizers['correcting'], 
                    scheduler= schedulers['correcting'], 
                    device= device
                )
                
                models['correcting'] = correcting_state['model']
                optimizers['correcting'] = correcting_state['optimizer'] 
                schedulers['correcting'] = correcting_state['scheduler']

                starting_epoch = min(primary_state['epoch'], correcting_state['epoch']) + 1

            else:
                starting_epoch = 1

            ancillary_state = load_checkpoint(
                checkpoint_path= checkpoint_path['ancillary'], 
                model= models['ancillary'],
                optimizer= optimizers['ancillary'], 
                scheduler= schedulers['ancillary'], 
                device= device
            )
            models['ancillary'] = ancillary_state['model']

        elif config.training['training_stage'] == 3:
            if config.training['continue_traning']:
                primary_state = load_checkpoint(
                    checkpoint_path= checkpoint_path['primary'], 
                    model= models['primary'],
                    optimizer= optimizers['primary'], 
                    scheduler= schedulers['primary'], 
                    device= device
                )
                starting_epoch = primary_state['epoch'] + 1
                models['primary'] = primary_state['model']
                optimizers['primary'] = primary_state['optimizer']
                schedulers['primary'] = primary_state['scheduler']

            else:
                primary_state = load_checkpoint(
                    checkpoint_path= checkpoint_path['primary'], 
                    model= models['primary'],
                    optimizer= optimizers['primary'], 
                    scheduler= schedulers['primary'], 
                    device= device
                )
                models['primary'] = primary_state['model']
                starting_epoch = 1

            correcting_state = load_checkpoint(
                checkpoint_path= checkpoint_path['correcting'], 
                model= models['correcting'],
                optimizer= optimizers['correcting'], 
                scheduler= schedulers['correcting'], 
                device= device
            )
            models['correcting'] = correcting_state['model']

            ancillary_state = load_checkpoint(
                checkpoint_path= checkpoint_path['ancillary'], 
                model= models['ancillary'],
                optimizer= optimizers['ancillary'], 
                scheduler= schedulers['ancillary'], 
                device= device
            )
            models['ancillary'] = ancillary_state['model']
    else:
        starting_epoch = 1
    
    save_dir = os.path.join(ROOT, config.data['output_path'], config.experiment['name'], config.experiment['version'])
    os.makedirs(save_dir, exist_ok=True)

    logger = Logger(save_dir)
    logger.info(f"Starting the experiment: {config.experiment['name']} {config.experiment['version']}")
    logger.info(f"Using {torch.cuda.device_count()} GPUs")
    logger.info(f"Using device: {device}")
    logger.info(f"Fully Supervised Training dataset size: {len(fully_sup_train_dataset)}")
    logger.info(f"Weak Training dataset size: {len(weak_train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")

    lrs = []
    logger.info(f"Starting training from epoch: {starting_epoch}")
    
    if config.training['training_stage'] == 1:
        stage1_training_loop(
            starting_epoch, config, train_loaders, val_loader, train_transform,
            val_transform, device, models, optimizers, schedulers, loss_funcs, 
            scaler, logger, save_dir
        )
        logger.info("Stage 1 Training finished successfully")

    elif config.training['training_stage'] == 2:
        stage2_training_loop(
            starting_epoch, config, train_loaders, val_loader, train_transform, 
            val_transform, device, models, optimizers, schedulers, loss_funcs, 
            scaler, logger, save_dir
        )
        logger.info("Stage 2 Training finished successfully")

    elif config.training['training_stage'] == 3:
        stage3_training_loop(
            starting_epoch, config, train_loaders, val_loader, train_transform, 
            val_transform,device, models, optimizers, schedulers, loss_funcs, 
            scaler, logger, save_dir
        )
        logger.info("Stage 3 Training finished successfully")

    else:
        stage1_training_loop(
            starting_epoch, config, train_loaders, val_loader, train_transform,
            val_transform, device, models, optimizers, schedulers, loss_funcs, 
            scaler, logger, save_dir
        )
        logger.info("Stage 1 Training finished successfully")

        stage2_training_loop(
            starting_epoch, config, train_loaders, val_loader, train_transform, 
            val_transform, device, models, optimizers, schedulers, loss_funcs, 
            scaler, logger, save_dir
        )
        logger.info("Stage 2 Training finished successfully")

        stage3_training_loop(
            starting_epoch, config, train_loaders, val_loader, train_transform, 
            val_transform,device, models, optimizers, schedulers, loss_funcs, 
            scaler, logger, save_dir
        )
        logger.info("Stage 3 Training finished successfully")

        logger.info("All the 3 stages are finished successfully")


if __name__ == "__main__":
    config = load_config(os.path.join(ROOT, "configs/config.yml"))
    train(config)

"""

DONE:
    1. Build the 3 models
    2. Initilize 2 optimizers
    3. Be sure about input consistency of (data_loaders, models, optimizers) across the 3 stages
    4. Before you build f_training_loader split f to f1 and f2 and 
        build a data_loader instance for each
    5. Build 3 function: one for each stage training loop
    6. optimizer, schedular etc parsing to the above 3 functions
    7. Write configuration yaml file in configs/

what to do next:

    8. Run on Kaggle
"""