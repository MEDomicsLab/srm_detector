"""
    @file:              predicts.py
    @Author:            Moustafa Amine Bezzahi , Ihssene Brahimi

    @Creation Date:     06/2024
    @Last modification: 08/2024

    @Description:       This file is used to define the prediction functions of the segmentation models.
"""

import os
import numpy as np
from glob import glob
import torch
from torch import nn
import argparse
import time
import monai
import nrrd

from ...utils import visualize_evaluation_srm
from monai.inferers import sliding_window_inference
from ...data.transforms import get_kidney_transforms

from ...constants import *

st = time.time()
num_classes = 3
model_folder = KIDNEY_SEGMENTATION_MODEL
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = monai.networks.nets.BasicUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=num_classes,
        features=(16, 32, 64, 128, 256, 32),
        dropout=0.1,
    )

# model = nn.DataParallel(model)
model.to(device)



def find_nrrd_files(root_dir):
    images = []
    #segments = []

    # Traverse through all subdirectories and files in the root directory
    for root, dirs, files in os.walk(root_dir):
        for dir_name in dirs:
            if dir_name == "CECT":
                cect_dir = os.path.join(root, dir_name)
                # Check if the CECT directory contains the required files
                image_file = os.path.join(cect_dir, "CECT.nrrd")
                #segment_file = os.path.join(cect_dir, "CECT_Kidneys_Segmentation.seg.nrrd")
                if os.path.exists(image_file):
                    images.append(image_file)

    return images 



# Specify the root directory containing patient folders
holdout_data_folder = '/projects/renal/DATA_CONFIDENTIAL/structured_data/unlabeled_test'

# Find NRRD files for images and segments
holdout_images = find_nrrd_files(holdout_data_folder)

# Print the list of images and segments
print("List of test image paths:")

print(len(holdout_images))


keys = ("img", "seg", "id")
holdout_images = sorted(holdout_images)

holdout_patient_folders = [folder for folder in os.listdir(holdout_data_folder)
                                if folder.startswith("Renal-CHUS-") and 
                                os.path.exists(os.path.join(holdout_data_folder, folder, "CECT"))] 

holdout_patient_folders = sorted(holdout_patient_folders)
print(f"training: total labeled image: ({len(holdout_patient_folders)}) from folder: {holdout_data_folder}")


holdout_files = [{keys[0]: img, keys[2]: id} for img, id in zip(holdout_images, holdout_patient_folders)]

# create a test data loader
holdout_transforms = get_kidney_transforms("unlabeled_train", keys[0])
holdout_ds = monai.data.CacheDataset(data=holdout_files, transform=holdout_transforms)
holdout_loader = monai.data.DataLoader(
    holdout_ds,
    batch_size=1,  # image-level batch to the sliding window method, not the window-level batch
    num_workers=2,
    pin_memory=False#torch.cuda.is_available(),
)


model.load_state_dict(torch.load(os.path.join("/projects/renal/Notebooks/Models/semisup/model/best_metric_model_full_256.pth")))
model.eval()

def test_sample_number(sample_num, test_loader, eva_path = "/projects/renal/Notebooks/Models/semisup/output"):
    labeled=False
    for i, test_data in enumerate(test_loader):
        image = test_data['img']
        if test_data['seg']:
            gt = test_data['seg']
            labeled=True
        roi_size = (128, 128, 16)
        sw_batch_size = 4
        test_outputs = sliding_window_inference(test_data["img"].to(device), roi_size, sw_batch_size, model)
        
        if i == sample_num:
            break  # take just i sample from test folder

    # Initialize empty arrays to store image and mask slices
    image_slices = []
    gt_slices = []
    mask_slices = []

    # Iterate over slices and append to lists
    for slice_idx in range(image.size(2)):  # Assuming 34 slices based on the size of the test output
        image_slice = image[0, 0, :, :, slice_idx].numpy()
        if labeled==True:
            gt_slice = gt[0, 0, :, :, slice_idx].numpy()
            gt_slices.append(gt_slice)
        mask_slice = torch.argmax(test_outputs, dim=1).detach().cpu()[0, :, :, slice_idx].numpy()

        image_slices.append(image_slice)
        
        mask_slices.append(mask_slice)

    # Convert lists to numpy arrays
    image_volume = np.array(image_slices)
    if labeled==True:
        gt_volume = np.array(gt_slices)
    mask_volume = np.array(mask_slices)

    # Call visualize_evaluation to save the GIF
    case_id = sample_num  # Assuming case_id is 1 for example
    eva_path = eva_path  # Output path for GIF
    if labeled==True:
        visualize_evaluation_srm(case_id, image_volume, gt_volume, mask_volume, eva_path)
    else:
        visualize_evaluation_srm(case_id, image_volume, mask_volume, mask_volume, eva_path)



test_sample_number(3, holdout_loader, eva_path = "/projects/renal/Notebooks/Models/semisup/output")



def save_predictions(holdout_loader):
    for i, holdout_data in enumerate(holdout_loader):
        image = holdout_data['img']
        #segment = holdout_data['seg']
        patient_id = holdout_data['id'][0]
        roi_size = (128, 128, 16)
        sw_batch_size = 4
        holdout_outputs = sliding_window_inference(holdout_data["img"].to(device), roi_size, sw_batch_size, model)

        # Convert predictions to numpy array
        holdout_outputs_np = torch.argmax(holdout_outputs, dim=1).detach().cpu().numpy()

        # Save each prediction as an NRRD file
        for batch_idx in range(holdout_outputs_np.shape[0]):
            prediction = holdout_outputs_np[batch_idx]
            prediction_filename = os.path.join(SEGMENTATION_OUTPUT_FILES, f'mt_prediction_{patient_id}.seg.nrrd')
            nrrd.write(prediction_filename, prediction)
            print(f'Patient {patient_id} saved!')