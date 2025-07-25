"""
    @file:              semi_supervised.py
    @Author:            Moustafa Amine Bezzahi, Ihssene Brahimi

    @Creation Date:     06/2024
    @Last modification: 07/2024

    @Description:       This file is used to define the semisupervised segmentation models.
"""
import numpy as np
import monai
import nrrd

from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch

import os
import torch
import torch.nn as nn
from time import time

import monai.transforms as mt
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
#from torch.utils.tensorboard import SummaryWriter

from constants import *
from utils import get_current_consistency_weight, update_ema_variables, dice_coef, iou

def kidney_segmentor(
            
            labeled_train_loader=None, 
            labeled_val_loader=None, 
            unlabeled_train_loader=None, 
            model_path=KIDNEY_SEGMENTATION_MODEL,
            device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
            num_classes=3, 
            features=(16, 32, 64, 128, 256, 32),
            teacher_dropout=0.1, 
            student_dropout=0.1, 
            lr= 1e-4, 
            patch_size = (128, 128, 16), # for original images. (32,32,16) for cropped images
            max_epochs=1000,
            MeanTeacherEpoch=10, 
            val_interval=1,
            consistency=5,
            consistency_rampup=250.0,
            patience=10,
            inference_mode=False,
            weights=None,     
            inference_loader=None, # Required for inference
            infer_output=None,
            ):

    
    # student model
    model = monai.networks.nets.BasicUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=num_classes, 
            features=features,  
            dropout=student_dropout,
        )

        
    
     # If inference_mode is enabled, skip training and perform inference
    if inference_mode:
        if weights is None:
            raise ValueError("Model weights must be provided for inference_mode.")
        if inference_loader is None:
            raise ValueError("Input data must be provided for inference_mode.")
        
        print("Running inference mode...")
        state_dict = torch.load(weights, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        # Ensure the predictions directory exists
        #predictions_dir = './inference_roi_predictions'
        os.makedirs(infer_output, exist_ok=True)

        # Evaluation loop
        for i, holdout_data in enumerate(inference_loader):
            image = holdout_data['img']
            segment = holdout_data['seg']
            if holdout_data['id'][0]:
                patient_id = holdout_data['id'][0]
            else:
                patient_id = i
            roi_size = (128, 128, 16)
            sw_batch_size = 1
            holdout_outputs = sliding_window_inference(holdout_data["img"].to(device), roi_size, sw_batch_size, model)
            
            # Convert predictions to numpy array
            holdout_outputs_np = torch.argmax(holdout_outputs, dim=1).detach().cpu().numpy()

            # Save each prediction as an NRRD file
            for batch_idx in range(holdout_outputs_np.shape[0]):
                prediction = holdout_outputs_np[batch_idx]
                prediction_filename = os.path.join(infer_output, f'mt_prediction.seg.nrrd')
                nrrd.write(prediction_filename, prediction)
                print(f'The ROI prediction of Patient {patient_id} saved!')
    
        return holdout_outputs  # 3D Prediction Map. Use torch.argmax(holdout_outputs, dim=1).detach().cpu()[0, :, :, axial_slice_number]

    # Teacher model definition (only used in training)
    ema_model = monai.networks.nets.BasicUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=num_classes,
            features=features,   
            dropout=teacher_dropout,
        )
    for param in ema_model.parameters():
        param.detach_()

    # A test DP training strategy
    model = nn.DataParallel(model) 
    ema_model = nn.DataParallel(ema_model)
    model.to(device)
    ema_model.to(device)

    
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    sw_batch_size = 4


    best_metric = -1
    best_metric_epoch = -1
    iter_num = 0
    best_loss = float('inf')
    epochs_no_improve = 0
    early_stop = False
    epoch_loss_values = []
    metric_values = []
    metric_values_kidney = []
    metric_values_tumor = []
    post_pred = mt.Compose([mt.EnsureType(), mt.AsDiscrete(argmax=True, to_onehot=num_classes)])
    post_label = mt.Compose([mt.EnsureType(), mt.AsDiscrete(to_onehot=num_classes)])

    #writer = SummaryWriter()

    for epoch in range(max_epochs):
        if early_stop:
            print("Early stopping triggered.")
            break
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        start_time = time()
        model.train()
        epoch_loss = 0
        step = 0

        for labeled_batch, unlabeled_batch in zip(labeled_train_loader, unlabeled_train_loader):
            step += 1
            labeled_inputs, labels = (
                labeled_batch["img"].to(device),
                labeled_batch["seg"].to(device),
            )
            unlabeled_inputs = unlabeled_batch["img"].to(device)

            opt.zero_grad()

            noise_labeled = torch.clamp(torch.randn_like(
                    labeled_inputs) * 0.1, -0.2, 0.2)
            noise_unlabeled = torch.clamp(torch.randn_like(
                    unlabeled_inputs) * 0.1, -0.2, 0.2)
            noise_labeled_inputs = labeled_inputs + noise_labeled
            noise_unlabeled_inputs = unlabeled_inputs + noise_unlabeled

            outputs = model(labeled_inputs)
            with torch.no_grad():
                soft_out = torch.softmax(outputs, dim=1)
                outputs_unlabeled = model(unlabeled_inputs)
                soft_unlabeled = torch.softmax(outputs_unlabeled, dim=1)
                outputs_aug = ema_model(noise_labeled_inputs)
                soft_aug = torch.softmax(outputs_aug, dim=1)
                outputs_unlabeled_aug = ema_model(noise_unlabeled_inputs)
                soft_unlabeled_aug = torch.softmax(outputs_unlabeled_aug, dim=1)

            supervised_loss = loss_function(outputs, labels)
            if epoch < MeanTeacherEpoch:
                    consistency_loss = 0.0
            else:
                consistency_loss = torch.mean(
                    (soft_out - soft_aug) ** 2) + \
                                torch.mean(
                    (soft_unlabeled - soft_unlabeled_aug) ** 2)
            consistency_weight = get_current_consistency_weight(iter_num//150, consistency, consistency_rampup)
            iter_num += 1

            loss = supervised_loss + consistency_weight * consistency_loss
            loss.backward()
            opt.step()
            update_ema_variables(model, ema_model, 0.99, iter_num)

            epoch_loss += loss.item()
            print(
                f"{step}/{len(unlabeled_train_loader.dataset) // unlabeled_train_loader.batch_size}, "
                f"train_loss: {loss.item():.4f}")
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                for val_data in labeled_val_loader:
                    val_inputs, val_labels = (
                        val_data["img"].to(device),
                        val_data["seg"].to(device),
                    )
                    val_outputs = sliding_window_inference(val_inputs, patch_size, sw_batch_size, model)
                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                    # compute metric for current iteration
                    dice_metric(y_pred=val_outputs, y=val_labels)

                # aggregate the final mean dice result
                metric = dice_metric.aggregate().item()
                print(f"val dice: {metric}")
                # write values on TF board
                metric_values.append(metric)
                #writer.add_scalar("dice_metric", metric, epoch)
                # get mean dice for kidney
                metric_kidney = metric[0].item()
                metric_values_kidney.append(metric_kidney)
                #writer.add_scalar("dice_metric_kidney", metric_kidney, epoch)
                # get mean dice for tumor
                metric_tumor = metric[1].item()
                metric_values_tumor.append(metric_tumor)
                #writer.add_scalar("dice_metric_tumor", metric_tumor, epoch)
                # Early stopping logic based on Dice metric
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.module.state_dict(), os.path.join(model_path, "best_mean_teacher_model.pth"))
                    print("Saved new best metric model")
                    
                    epochs_no_improve = 0  # Reset counter when improvement is seen
                else:
                    epochs_no_improve += 1
                
                # Early stopping logic based on loss
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    early_stop = True
                    break
                print(
                    f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                    f"\nbest mean dice: {best_metric:.4f} "
                    f"at epoch: {best_metric_epoch}"
                )
                # reset the status for next validation round
                dice_metric.reset()

                # metric_values.append(metric)
                # if metric > best_metric:
                #     best_metric = metric
                #     best_metric_epoch = epoch + 1
                #     torch.save(model.module.state_dict(), os.path.join(
                #         model_path, "best_mean_teacher_model.pth"))
                #     print("saved new best metric model")
                
            print(f"epoch time = {time() - start_time}")

    #writer.close()

    return val_interval, epoch_loss_values, metric_values, metric_values_kidney, metric_values_tumor


def srm_segmentor(
            
            labeled_train_loader, 
            labeled_val_loader, 
            unlabeled_train_loader, 
            model_path=SRM_SEGMENTATION_MODEL,
            device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
            num_classes=2, 
            features=(8, 16, 32, 64, 128, 16),
            teacher_dropout=0.1, 
            student_dropout=0.1, 
            lr= 1e-4, 
            patch_size = (32, 32, 16), # for original images. (32,32,16) for cropped images
            max_epochs=1000,
            MeanTeacherEpoch=10, 
            val_interval=1,
            consistency=5,
            consistency_rampup=250.0,
            patience=10,
            weights=None,

            ):

    
    # student model
    model = monai.networks.nets.BasicUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=num_classes,
            features=features,  
            dropout=student_dropout,
        )

    # teacher model
    ema_model = monai.networks.nets.BasicUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=num_classes,
            features=features,   
            dropout=teacher_dropout,
        )
    for param in ema_model.parameters():
        param.detach_()

    # A test DP training strategy
    model = nn.DataParallel(model) 
    ema_model = nn.DataParallel(ema_model)
    model.to(device)
    ema_model.to(device)

    
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    sw_batch_size = 4


    best_metric = -1
    best_metric_epoch = -1
    iter_num = 0
    epoch_loss_values = []
    best_loss = float('inf')
    epochs_no_improve = 0
    early_stop = False
    metric_values = []
    post_pred = mt.Compose([mt.EnsureType(), mt.AsDiscrete(argmax=True, to_onehot=num_classes)])
    post_label = mt.Compose([mt.EnsureType(), mt.AsDiscrete(to_onehot=num_classes)])

    #writer = SummaryWriter()

    for epoch in range(max_epochs):
        if early_stop:
            print("Early stopping triggered.")
            break
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        start_time = time()
        model.train()
        epoch_loss = 0
        step = 0

        for labeled_batch, unlabeled_batch in zip(labeled_train_loader, unlabeled_train_loader):
            step += 1
            labeled_inputs, labels = (
                labeled_batch["img"].to(device),
                labeled_batch["seg"].to(device),
            )
            unlabeled_inputs = unlabeled_batch["img"].to(device)

            opt.zero_grad()

            noise_labeled = torch.clamp(torch.randn_like(
                    labeled_inputs) * 0.1, -0.2, 0.2)
            noise_unlabeled = torch.clamp(torch.randn_like(
                    unlabeled_inputs) * 0.1, -0.2, 0.2)
            noise_labeled_inputs = labeled_inputs + noise_labeled
            noise_unlabeled_inputs = unlabeled_inputs + noise_unlabeled

            outputs = model(labeled_inputs)
            with torch.no_grad():
                soft_out = torch.softmax(outputs, dim=1)
                outputs_unlabeled = model(unlabeled_inputs)
                soft_unlabeled = torch.softmax(outputs_unlabeled, dim=1)
                outputs_aug = ema_model(noise_labeled_inputs)
                soft_aug = torch.softmax(outputs_aug, dim=1)
                outputs_unlabeled_aug = ema_model(noise_unlabeled_inputs)
                soft_unlabeled_aug = torch.softmax(outputs_unlabeled_aug, dim=1)

            supervised_loss = loss_function(outputs, labels)
            if epoch < MeanTeacherEpoch:
                    consistency_loss = 0.0
            else:
                consistency_loss = torch.mean(
                    (soft_out - soft_aug) ** 2) + \
                                torch.mean(
                    (soft_unlabeled - soft_unlabeled_aug) ** 2)
                
            consistency_weight = get_current_consistency_weight(iter_num//150, consistency, consistency_rampup)
            iter_num += 1

            loss = supervised_loss + consistency_weight * consistency_loss
            loss.backward()
            opt.step()
            update_ema_variables(model, ema_model, 0.99, iter_num)

            epoch_loss += loss.item()
            print(
                f"{step}/{len(unlabeled_train_loader.dataset) // unlabeled_train_loader.batch_size}, "
                f"train_loss: {loss.item():.4f}")
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                for val_data in labeled_val_loader:
                    val_inputs, val_labels = (
                        val_data["img"].to(device),
                        val_data["seg"].to(device),
                    )
                    val_outputs = sliding_window_inference(val_inputs, patch_size, sw_batch_size, model)
                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                    # compute metric for current iteration
                    dice_metric(y_pred=val_outputs, y=val_labels)

                # aggregate the final mean dice result
                metric = dice_metric.aggregate().item()
                print(f"val dice: {metric}")
                # write values on TF board
                metric_values.append(metric)
                #writer.add_scalar("dice_metric", metric, epoch)
                # Early stopping logic based on Dice metric
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.module.state_dict(), os.path.join(model_path, "best_tumor_mean_teacher_model.pth"))
                    print("Saved new best metric model")
                    epochs_no_improve = 0  # Reset counter when improvement is seen
                else:
                    epochs_no_improve += 1
                
                # Early stopping logic based on loss
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    early_stop = True
                    break
                print(
                    f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                    f"\nbest mean dice: {best_metric:.4f} "
                    f"at epoch: {best_metric_epoch}"
                )
                # reset the status for next validation round
                dice_metric.reset()

                # metric_values.append(metric)
                # if metric > best_metric:
                #     best_metric = metric
                #     best_metric_epoch = epoch + 1
                #     torch.save(model.module.state_dict(), os.path.join(
                #         model_path, "best_mean_teacher_model.pth"))
                #     print("saved new best metric model")
                
            print(f"epoch time = {time() - start_time}")

    #writer.close()

    return val_interval, epoch_loss_values, metric_values

            