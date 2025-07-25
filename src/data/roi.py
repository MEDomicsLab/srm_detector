"""
    @file:              roi.py
    @Author:            Ihssene Brahimi, Moustafa Amine Bezzahi

    @Creation Date:     05/2024
    @Last modification: 07/2024

    @Description:       This file is used to define ROIs in the imaging dataset.
"""

import monai.transforms as mt
import os
from scipy.ndimage import label as lb
from scipy.ndimage import find_objects
from tqdm import tqdm
import pandas as pd
import ast 
from constants import *


def load_inference_data(patients_folder):
    data = {'root': [], 'img': [], 'seg': [], 'pred': [], 'bbox': []}
    # Loop through patients
    for patient_folder in tqdm(os.listdir(patients_folder), desc="Processing Labeled Patients"):
            
            data['root'].append(patient_folder)
            patient_path = os.path.join(patients_folder, patient_folder)
            if os.path.isdir(patient_path):
                cect_folder = os.path.join(patient_path, "CECT")
                    
                # Check if CECT folder exists
                if os.path.exists(cect_folder):
                    # Find segmentation and image files
                    pred_file = [f for f in os.listdir(cect_folder) if f.endswith("mt_prediction.seg.nrrd")]
                    img_file = [f for f in os.listdir(cect_folder) if f.endswith("CECT.nrrd")]
                    seg_file = [f for f in os.listdir(cect_folder) if f.endswith("CECT_Segmentation.seg.nrrd")]
                    if img_file and pred_file and seg_file:
                        pred_path = os.path.join(cect_folder, pred_file[0])
                        data['pred'].append(pred_path)
                        img_path = os.path.join(cect_folder, img_file[0])
                        data['img'].append(img_path)
                        seg_path = os.path.join(cect_folder, seg_file[0])
                        data['seg'].append(seg_path)

   
    
    return data

def load_labeled_data(patients_folder=LABELED_DATA_PATH):
    data = {'root': [], 'img': [], 'seg': [], 'pred': [], 'bbox': []}
    # Loop through patients
    for patient_folder in tqdm(os.listdir(patients_folder), desc="Processing Labeled Patients"):
            
            data['root'].append(patient_folder)
            patient_path = os.path.join(patients_folder, patient_folder)
            if os.path.isdir(patient_path):
                cect_folder = os.path.join(patient_path, "CECT")
                    
                # Check if CECT folder exists
                if os.path.exists(cect_folder):
                    # Find segmentation and image files
                    pred_file = [f for f in os.listdir(cect_folder) if f == "CECT_Kidneys_Segmentation.seg.nrrd"]
                    seg_file = [f for f in os.listdir(cect_folder) if f.endswith("CECT_Segmentation.seg.nrrd")]
                    img_file = [f for f in os.listdir(cect_folder) if f.endswith("CECT.nrrd")]

                    if seg_file and img_file and pred_file:
                        pred_path = os.path.join(cect_folder, pred_file[0])
                        data['pred'].append(pred_path)
                        seg_path = os.path.join(cect_folder, seg_file[0])
                        data['seg'].append(seg_path)
                        img_path = os.path.join(cect_folder, img_file[0])
                        data['img'].append(img_path)
    return data

def load_unlabeled_data(patients_folder=UNLABELED_DATA_PATH):
    data = {'root': [], 'img': [], 'seg': [], 'pred': [], 'bbox': []}
    # Loop through patients
    for patient_folder in tqdm(os.listdir(patients_folder), desc="Processing Unlabeled Patients"):
            
            data['root'].append(patient_folder)
            patient_path = os.path.join(patients_folder, patient_folder)
            mask_file = os.path.join(patient_path, "mt_prediction.seg.nrrd")
            data['pred'].append(mask_file)
            if os.path.isdir(patient_path):
                cect_folder = os.path.join(patient_path, "CECT")
                    
                # Check if CECT folder exists
                if os.path.exists(cect_folder):
                    # Find segmentation and image files
                    seg_file = [f for f in os.listdir(cect_folder) if f.endswith("CECT_Segmentation.seg.nrrd")]  # or f.endswith("CECT_Kidneys_Segmentation.seg.nrrd") 
                    img_file = [f for f in os.listdir(cect_folder) if f.endswith("CECT.nrrd")]

                    if seg_file and img_file:
                        seg_path = os.path.join(cect_folder, seg_file[0])
                        data['seg'].append(seg_path)
                        img_path = os.path.join(cect_folder, img_file[0])
                        data['img'].append(img_path)
    return data

# Function to calculate bounding boxes from binary masks
def bounding_box(mask):
    mask_np = mask.squeeze().cpu().numpy()  # Convert tensor to numpy array
    labeled, num_features = lb(mask_np)  # Label connected components
    objects = find_objects(labeled)  # Find bounding boxes for each connected component

    boxes = []
    for obj in objects:
        z_start, z_end = obj[0].start, obj[0].stop
        y_start, y_end = obj[1].start, obj[1].stop
        x_start, x_end = obj[2].start, obj[2].stop
        boxes.append((x_start, y_start, z_start, x_end - x_start, y_end - y_start, z_end - z_start))
    
    return boxes



def load_bbox(data, post_pred_transform): 
    
    for i in tqdm(range(len(data['root'])), desc="Loading boxes"):
        pred_sample = post_pred_transform(data['pred'][i])          
        # Calculate bounding boxes for ground truth mask and predicted mask
        pred_boxes = bounding_box(pred_sample)
        data['bbox'].append(pred_boxes)

    return data


def get_areas(df, data_name):
    
    labeled_test_df = pd.DataFrame(df)
    possible_areas = []
    dimensions = []
    # Loop through each row and each tuple to print the coordinates
    for index, row in labeled_test_df.iterrows():
        print(f"Patient {labeled_test_df['root'][index]}:")
        if isinstance(row['bbox'], str):
            row['bbox'] = ast.literal_eval(row['bbox'])

        for box in row['bbox']:
            x, y, z = box[3], box[4], box[5]
            area = x*y*z
            print(f"box: {box}, area: {area}, x: {x}, y: {y}, z: {z} ")
            possible_areas.append(area)
            dimensions.append((x,y,z))
        print()  # Blank line for better readability between rows

    print(f"There is a total of {len(possible_areas)} different box areas in the {data_name} set")

    return possible_areas, dimensions



def crop(resized_img, resized_seg, pred, box):

    
    x_start, y_start, z_start, x_len, y_len, z_len = box 


    x_center = x_start+x_len//2
    y_center = y_start+y_len//2
    z_center = z_start+z_len//2

    roi_center = (z_center, y_center, x_center)
    roi_size = (z_len,y_len,x_len)

    roi_area = z_len*y_len*x_len
            
    if roi_area > 1:  # skip tiny tumor boxes 
        
        # Crop the image, seg and prediction to match box area
        cropper = mt.SpatialCrop(roi_center=roi_center, roi_size=roi_size)
        
        
        crop_pred = cropper(pred)
        crop_img = cropper(resized_img)
        crop_seg = cropper(resized_seg)
        crop_depth = crop_pred.shape[3]

    return crop_img, crop_seg, crop_pred, crop_depth

def load_resized_data(data, i):

    # Load image, segmentation, prediction and boxes from data.csvseg_path = data["seg"][i]
    img_path = data['img'][i]
    pred_path = data['pred'][i]
    seg_path = data['seg'][i]
    img_proc = mt.Compose([
            mt.LoadImage(image_only=True, ensure_channel_first=True, simple_keys=True),
            mt.EnsureType(), 
            mt.Orientation(axcodes='LPS'),
            mt.Spacing(pixdim=(2.0, 2.0, 5.0), mode=("nearest")), 
            mt.ScaleIntensityRange(a_min=-500.0, a_max=500.0, b_min=0.0, b_max=1.0, clip=True),
            mt.ToTensor()
            ])
    
    seg_proc = mt.Compose([
            mt.LoadImage(image_only=True, ensure_channel_first=True, simple_keys=True),
            mt.EnsureType(), 
            mt.Orientation(axcodes='LPS'),
            mt.Spacing(pixdim=(2.0, 2.0, 5.0), mode=("bilinear")), 
            mt.ScaleIntensityRange(a_min=-500.0, a_max=500.0, b_min=0.0, b_max=1.0, clip=True),
            mt.ToTensor()
            ])

    pred_proc = mt.Compose([
        mt.LoadImage(image_only=True, ensure_channel_first=True, simple_keys=True),
        mt.EnsureType(), 
        mt.Orientation(axcodes='LPS'),
        mt.ScaleIntensityRange(a_min=-500.0, a_max=500.0, b_min=0.0, b_max=1.0, clip=True),
        mt.ToTensor()
        ])
    
    seg = seg_proc(seg_path)
    img = img_proc(img_path)
    pred = pred_proc(pred_path)

    # sitk : D H W    monai : W H D
    x = pred.shape[1]
    y = pred.shape[2]
    z = pred.shape[3]

    # Resize image and segmentation so they match prediction in shape
    resize_transform = mt.Resize(spatial_size=(x, y, z))
    resized_img = resize_transform(img)
    resized_seg = resize_transform(seg)

    return resized_img, resized_seg, pred