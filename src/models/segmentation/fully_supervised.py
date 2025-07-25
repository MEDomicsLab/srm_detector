"""
    @file:              fully_supervised.py
    @Author:            Moustafa Amine Bezzahi, Ihssene Brahimi

    @Creation Date:     06/2024
    @Last modification: 08/2024

    @Description:       This file is used to define the fully supervised segmentation model trainer.
"""

from monai.networks.layers import Norm
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
import torch
import monai
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from monai.transforms import Compose, EnsureType, AsDiscrete
from constants import *


def trainer(train_loader, 
            val_loader,
            model_path=SRM_SEGMENTATION_MODEL,
            device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
            num_classes=2,
            dropout=0.5,
            features=(16, 32, 64, 128, 256, 32),
            lr= 1e-4,
            roi_size = (32, 32, 16), # for original images. (32,32,16) for cropped images
            max_epochs=500,
            val_interval=2
            ):
    
    model = monai.networks.nets.BasicUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=num_classes,
            features=features,
            dropout=dropout,
        )
    

    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-4)
    dice_metric = DiceMetric(include_background=False, reduction="mean")   


    # Training parameters
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []

    model = model.to(device)

    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    writer = SummaryWriter()

    sw_batch_size = 4
    post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=num_classes)])
    post_label = Compose([EnsureType(), AsDiscrete(to_onehot=num_classes)])


    for epoch in range(max_epochs):
            print("-" * 10)
            print(f"epoch {epoch + 1}/{max_epochs}")
            model.train()
            epoch_loss = 0
            step = 0
            for batch_data in train_loader:
                step += 1
                inputs, labels = batch_data["img"].to(device), batch_data["seg"].to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                epoch_len = len(train_loader)
                print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
                writer.add_scalar(f"train_loss", loss.item(), epoch_len * epoch + step)
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

            # Validation
            if (epoch + 1) % val_interval == 0:
                model.eval()
                with torch.no_grad():
                    val_loss = 0
                    for val_data in val_loader:
                        val_images, val_labels = val_data["img"].to(device), val_data["seg"].to(device)
                        val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
                        val_loss += loss_function(val_outputs, val_labels).item()
                        val_outputs = decollate_batch(val_outputs)
                        val_labels = decollate_batch(val_labels)
                        val_outputs = [post_pred(i) for i in val_outputs]
                        val_labels = [post_label(i) for i in val_labels]
                        dice_metric(y_pred=val_outputs, y=val_labels)
                    val_loss /= len(val_loader)
                    metric = dice_metric.aggregate().item()
                    dice_metric.reset()
                    metric_values.append(metric)
                    print(f"epoch {epoch + 1} validation loss: {val_loss:.4f}, validation dice: {metric:.4f}")
                    writer.add_scalar(f"val_loss", val_loss, epoch + 1)
                    writer.add_scalar(f"val_dice", metric, epoch + 1)

                    if metric > best_metric:
                        best_metric = metric
                        best_metric_epoch = epoch + 1
                        torch.save(model.state_dict(), os.join.path(model_path, f"best_srm_fully_supervised_model.pth"))
                        print("saved new best metric model")

            # Save model checkpoint for every epoch
            '''checkpoint_path = f"/projects/renal/03_tumour_segmentation/checkpoints/basic_unet_checkpoint_epoch_{epoch + 1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, checkpoint_path)'''

            scheduler.step()

    print(f"Train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()
