"""
    @file:              dataset.py
    @Author:            Ihssene Brahimi, Moustafa Amine Bezzahi

    @Creation Date:     05/2024
    @Last modification: 07/2024

    @Description:       This file is used to define clinical and imaging data structure.
"""

import numpy as np
import ast
import torch
from torch.utils.data import Dataset
import monai.transforms as mt
from scipy.ndimage import label, find_objects
from ..utils import extract_features



class ROIDataset(Dataset):
    """
    Return cropped dataset for segmentation task as one batch with patient ID, image and seg.
    """
    def __init__(self, data, q3_x_len=30, q3_y_len=61, q3_z_len=45, transform=None):
        self.data = data
        self.q3_x_len = q3_x_len
        self.q3_y_len = q3_y_len
        self.q3_z_len = q3_z_len
        self.transform = transform

    def __len__(self):
        return len(self.data['root'])

    def get_box_centers(self, box):
        x_start, y_start, z_start, x_len, y_len, z_len = box
        x_center = x_start + x_len // 2
        y_center = y_start + y_len // 2 
        z_center = z_start + z_len // 2 
        return x_center, y_center, z_center

    def crop(self, resized_img, resized_seg, pred, box):
        
        x_start, y_start, z_start, x_len, y_len, z_len = box 

        x_center = x_start+x_len//2
        y_center = y_start+y_len//2
        z_center = z_start+z_len//2

        roi_center = (z_center, y_center, x_center)
        roi_size = (z_len,y_len,x_len)

            
        # Crop the image, seg and prediction to match box area
        cropper = mt.SpatialCrop(roi_center=roi_center, roi_size=roi_size)
        
        
        crop_pred = cropper(pred)
        crop_img = cropper(resized_img)
        crop_seg = cropper(resized_seg)
        crop_depth = crop_pred.shape[3]

        return crop_img, crop_seg, crop_pred, crop_depth


    def load_resized_data(self, data, i):

        # Load image, segmentation, prediction and boxes from data.csv seg_path = data["seg"][i]
        img_path = data['img'][i]
        pred_path = data['pred'][i]
        seg_path = data['seg'][i]
        img_proc = mt.Compose([
                mt.LoadImage(image_only=True, ensure_channel_first=True),
                mt.EnsureType(), 
                mt.Orientation(axcodes='LPS'),
                mt.Spacing(pixdim=(2.0, 2.0, 5.0), mode=("nearest")), 
                mt.ScaleIntensityRange(a_min=-500.0, a_max=500.0, b_min=0.0, b_max=1.0, clip=True),
                mt.ToTensor()
                ])
        
        seg_proc = mt.Compose([
                mt.LoadImage(image_only=True, ensure_channel_first=True),
                mt.EnsureType(), 
                mt.Orientation(axcodes='LPS'),
                mt.Spacing(pixdim=(2.0, 2.0, 5.0), mode=("nearest")), 
                mt.ToTensor()
                ])

        pred_proc = mt.Compose([
                    mt.LoadImage(image_only=True, ensure_channel_first=True, simple_keys=True),
                    mt.Orientation(axcodes='LPS'),
                    mt.LabelToContour(),
                    mt.KeepLargestConnectedComponent(is_onehot=False, independent=True, connectivity=3, num_components=2),
                    mt.FillHoles(applied_labels=[1,2], connectivity=3),
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
        resize_transform_seg = mt.Resize(spatial_size=(x, y, z), mode="nearest")
        resize_transform_img = mt.Resize(spatial_size=(x, y, z))
        resized_img = resize_transform_img(img)
        resized_seg = resize_transform_seg(seg)


        return resized_img, resized_seg, pred
    
    def resize_before_train(self, img, seg):


        resize_transform_seg = mt.Resize(spatial_size=(64, 64, 32), mode="nearest")
        resize_transform_img = mt.Resize(spatial_size=(64, 64, 32))
        resized_img = resize_transform_img(img)
        resized_seg = resize_transform_seg(seg)
        

        return resized_img, resized_seg
    
        # Define the checker function
        

    def checker(self, crop_seg):
        """
        Check if the crop_seg contains a tumor.
        
        Parameters:
        crop_seg (numpy array): The cropped segmentation mask.
        box (tuple): The bounding box coordinates (x, y, z, x_len, y_len, z_len).
        
        Returns:
        int: 1 if the crop_seg contains a tumor, 0 otherwise.
        """
        # Tumor label is 1
        
        tumor_label = 0.5
        return 1 if np.any(crop_seg > tumor_label) else 0
        

    def retrieve_kidney_boxes(self, boxes):
        
        kidney_boxes = []
        for box in boxes:
            x_start, y_start, z_start, x_len, y_len, z_len = box 
            volume = x_len * y_len * z_len
            kidney_boxes.append((x_start, y_start, z_start, x_len, y_len, z_len, volume))
        kidney_boxes = sorted(kidney_boxes, key=lambda x: x[6], reverse=True)[:2]
        kidney_boxes = [(x_start, y_start, z_start, x_len, y_len, z_len) for x_start, y_start, z_start, x_len, y_len, z_len, _ in kidney_boxes]
        return kidney_boxes
    
    # Function to calculate bounding boxes from binary masks
    def bounding_box(self, mask):
        mask_np = mask.squeeze().cpu().numpy()  # Convert tensor to numpy array
        labeled, num_features = label(mask_np)  # Label connected components
        objects = find_objects(labeled)  # Find bounding boxes for each connected component

        boxes = []
        for obj in objects:
            z_start, z_end = obj[0].start, obj[0].stop
            y_start, y_end = obj[1].start, obj[1].stop
            x_start, x_end = obj[2].start, obj[2].stop
            x_len = x_end - x_start
            y_len = y_end - y_start
            z_len = z_end - z_start
            volume = x_len * y_len * z_len
            boxes.append((x_start, y_start, z_start, x_len, y_len, z_len, volume))
        
        # Sort boxes by volume in descending order and take the two largest
        boxes = sorted(boxes, key=lambda x: x[6], reverse=True)[:2]
        
        # Remove the volume information from the output
        boxes = [(x_start, y_start, z_start, x_len, y_len, z_len) for x_start, y_start, z_start, x_len, y_len, z_len, _ in boxes]
        
        return boxes


    def __getitem__(self, idx):
        
        patient_id = self.data['root'][idx]

        resized_img, resized_seg, pred = self.load_resized_data(self.data, idx)
        kidney_boxes = self.bounding_box(pred)

        r_box= kidney_boxes[0]
        l_box= kidney_boxes[1]

        l_crop_img, l_crop_seg, l_crop_pred, _ = self.crop(resized_img, resized_seg, pred, l_box)
        r_crop_img, r_crop_seg, r_crop_pred, _ = self.crop(resized_img, resized_seg, pred, r_box)
                            

        # select the kidney image and segment that contain tumor, ignore the other
        if self.checker(r_crop_seg) == 1:  # right contains tumor
            if self.transform:
                r_crop_img = self.transform[0](r_crop_img)
                r_crop_seg = self.transform[1](r_crop_seg)

            img, seg = self.resize_before_train(r_crop_img, r_crop_seg)
   

        else: # self.checker(l_crop_seg) == 1:  # left contains tumor
                if self.transform:
                    l_crop_img = self.transform[0](l_crop_img)
                    l_crop_seg = self.transform[1](l_crop_seg)
            
                img, seg = self.resize_before_train(l_crop_img, l_crop_seg)


        batch = {
            'Patient_ID': patient_id,
            'img': img,
            'seg': seg,
        }

        return batch
        

        

class SRMDataset(Dataset):
    """
     Used in the function "pos_neg_aug_datasets" to generate a positive/negative dataset 
     that will be used to control the original batch and augmented batch independently.
     It returns batches with patient ID, image, seg, label.
    """
    def __init__(self, data, data_rcc, q3_x_len=30, q3_y_len=61, q3_z_len=45, transform=None, augment_neg=False, augment_pos=False, aug_pos_rate=None, aug_neg_rate=None):
        self.data = data  # csv file that has patient id, image, seg, pred paths and bbox coords
        self.data_rcc = data_rcc
        self.q3_x_len = q3_x_len
        self.q3_y_len = q3_y_len
        self.q3_z_len = q3_z_len
        self.transform = transform
        self.augment_neg = augment_neg
        self.augment_pos = augment_pos
        self.aug_pos_rate = aug_pos_rate
        self.aug_neg_rate = aug_neg_rate

    def __len__(self):
        if "root" in self.data.keys():
            return len(self.data['root'])  # change to: data.keys()[0]
        else: 
            return len(self.data['img'])

    def get_box_centers(self, box):
        x_start, y_start, z_start, x_len, y_len, z_len = box
        x_center = x_start + x_len // 2
        y_center = y_start + y_len // 2 
        z_center = z_start + z_len // 2 
        return x_center, y_center, z_center

    def crop(self, resized_img, resized_seg, pred, box):
        
        x_start, y_start, z_start, x_len, y_len, z_len = box 

        x_center = x_start+x_len//2
        y_center = y_start+y_len//2
        z_center = z_start+z_len//2

        roi_center = (z_center, y_center, x_center)
        roi_size = (z_len,y_len,x_len)

            
        # Crop the image, seg and prediction to match box area
        cropper = mt.SpatialCrop(roi_center=roi_center, roi_size=roi_size)
        
        
        crop_pred = cropper(pred)
        crop_img = cropper(resized_img)
        crop_seg = cropper(resized_seg)
        crop_depth = crop_pred.shape[3]

        return crop_img, crop_seg, crop_pred, crop_depth


    def load_resized_data(self, data, i):

        # Load image, segmentation, prediction and boxes from data.csvseg_path = data["seg"][i]
        img_path = data['img'][i]
        pred_path = data['pred'][i]
        seg_path = data['seg'][i]
        img_proc = mt.Compose([
                mt.LoadImage(image_only=True, ensure_channel_first=True),
                mt.EnsureType(), 
                #mt.Orientation(axcodes='LPS'),
                mt.Spacing(pixdim=(2.0, 2.0, 5.0), mode=("nearest")), 
                mt.ScaleIntensityRange(a_min=-500.0, a_max=500.0, b_min=0.0, b_max=1.0, clip=True),
                mt.ToTensor()
                ])
        
        seg_proc = mt.Compose([
                mt.LoadImage(image_only=True, ensure_channel_first=True),
                mt.EnsureType(), 
                #mt.Orientation(axcodes='LPS'),
                mt.Spacing(pixdim=(2.0, 2.0, 5.0), mode=("bilinear")), 
                mt.ScaleIntensityRange(a_min=-500.0, a_max=500.0, b_min=0.0, b_max=1.0, clip=True),
                mt.ToTensor()
                ])

        pred_proc = mt.Compose([
                    mt.LoadImage(image_only=True, ensure_channel_first=True, simple_keys=True),
                    #mt.Orientation(axcodes='LPS'),
                    mt.LabelToContour(),
                    mt.KeepLargestConnectedComponent(is_onehot=False, independent=True, connectivity=3, num_components=2),
                    mt.FillHoles(applied_labels=[1,2], connectivity=3),
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
    
    def resize_before_train(self, img, seg):


        resize_transform = mt.Resize(spatial_size=(64, 64, 32))
        resized_img = resize_transform(img)
        resized_seg = resize_transform(seg)
        

        return resized_img, resized_seg
    

    def augmenter(self, img, seg, num_augmentations=3):

        transforms = mt.Compose([
            
            mt.RandFlip(spatial_axis=0, prob=0.5),
            mt.RandFlip(spatial_axis=1, prob=0.5),
            mt.RandFlip(spatial_axis=2, prob=0.5),
            mt.RandRotate(range_x=np.pi/4, prob=0.5),
            mt.RandAffine(prob=0.5, translate_range=(10, 10, 10)),
            mt.RandGaussianNoise(prob=0.5, mean=0.0, std=0.1), 
            mt.RandAdjustContrast(prob=0.5, gamma=(0.9, 1.1)),
            mt.RandShiftIntensity(offsets=0.5, prob=1.0),
        ])
        
        
        augmented_images = []
        augmented_segs = []
        for _ in range(num_augmentations):
            augmented_img = transforms(img)
            augmented_seg = transforms(seg)
            augmented_images.append(augmented_img)
            augmented_segs.append(augmented_seg)
        
        return augmented_images, augmented_segs
        

    def checker(self, crop_seg):
        """
        Check if the crop_seg contains a tumor.
        
        Parameters:
        crop_seg (numpy array): The cropped segmentation mask.
        box (tuple): The bounding box coordinates (x, y, z, x_len, y_len, z_len).
        
        Returns:
        int: 1 if the crop_seg contains a tumor, 0 otherwise.
        """
        # Tumor label is 1
        
        tumor_label = 0.5
        return 1 if np.any(crop_seg > tumor_label) else 0
        

    def retrieve_kidney_boxes(self, boxes):
        
        kidney_boxes = []
        for box in boxes:
            x_start, y_start, z_start, x_len, y_len, z_len = box 
            volume = x_len * y_len * z_len
            kidney_boxes.append((x_start, y_start, z_start, x_len, y_len, z_len, volume))
        kidney_boxes = sorted(kidney_boxes, key=lambda x: x[6], reverse=True)[:2]
        kidney_boxes = [(x_start, y_start, z_start, x_len, y_len, z_len) for x_start, y_start, z_start, x_len, y_len, z_len, _ in kidney_boxes]
        return kidney_boxes
    
    # Function to calculate bounding boxes from binary masks
    def bounding_box(self, mask):
        mask_np = mask.squeeze().cpu().numpy()  # Convert tensor to numpy array
        labeled, num_features = label(mask_np)  # Label connected components
        objects = find_objects(labeled)  # Find bounding boxes for each connected component

        boxes = []
        for obj in objects:
            z_start, z_end = obj[0].start, obj[0].stop
            y_start, y_end = obj[1].start, obj[1].stop
            x_start, x_end = obj[2].start, obj[2].stop
            x_len = x_end - x_start
            y_len = y_end - y_start
            z_len = z_end - z_start
            volume = x_len * y_len * z_len
            boxes.append((x_start, y_start, z_start, x_len, y_len, z_len, volume))
        
        # Sort boxes by volume in descending order and take the two largest
        boxes = sorted(boxes, key=lambda x: x[6], reverse=True)[:2]
        
        # Remove the volume information from the output
        boxes = [(x_start, y_start, z_start, x_len, y_len, z_len) for x_start, y_start, z_start, x_len, y_len, z_len, _ in boxes]
        
        return boxes


    def __getitem__(self, idx):
        
        patient_id = self.data['root'][idx]

        target = self.data_rcc[idx]
        resized_img, resized_seg, pred = self.load_resized_data(self.data, idx)
        kidney_boxes = self.bounding_box(pred)

        r_box= kidney_boxes[0]
        l_box= kidney_boxes[1]

        l_crop_img, l_crop_seg, l_crop_pred, _ = self.crop(resized_img, resized_seg, pred, l_box)
        r_crop_img, r_crop_seg, r_crop_pred, _ = self.crop(resized_img, resized_seg, pred, r_box)
                            

        # select the kidney image and segment that contain tumor, ignore the other
        if self.checker(r_crop_seg) == 1:  # right contains tumor
            if self.transform:
                r_crop_img = self.transform(r_crop_img)
                r_crop_seg = self.transform(r_crop_seg)

            img, seg = self.resize_before_train(r_crop_img, r_crop_seg)
        

        else: # self.checker(l_crop_seg) == 1:  # left contains tumor
                if self.transform:
                    l_crop_img = self.transform(l_crop_img)
                    l_crop_seg = self.transform(l_crop_seg)
            
                img, seg = self.resize_before_train(l_crop_img, l_crop_seg)

    

        batch = {
            'Patient_ID': patient_id,
            'img': img,
            'seg': seg,
            'pred': pred,
            'bbox': self.bounding_box(pred),
            'radiomics': extract_features(img),
            'label': torch.tensor(target, dtype=torch.long)  # Target is a single label
        }

        if self.augment_neg: 
            aug_img, aug_seg = self.augmenter(img, seg, self.aug_neg_rate)  

            aug_batch = {
                'Patient_ID': patient_id,
                'img': aug_img,
                'seg': aug_seg,
                'pred': pred,
                'bbox': self.bounding_box(pred),
                'radiomics': extract_features(aug_img),
                'label': torch.tensor(target, dtype=torch.long)  # Target is a single label
            }

            return batch, aug_batch
        
        elif self.augment_pos:
            aug_img, aug_seg = self.augmenter(img, seg, self.aug_pos_rate) 

            aug_batch = {
                'Patient_ID': patient_id,
                'img': aug_img,
                'seg': aug_seg,
                'pred': pred,
                'bbox': self.bounding_box(pred),
                'radiomics': extract_features(aug_img),
                'label': torch.tensor(target, dtype=torch.long)  # Target is a single label
            }

            return batch, aug_batch
        else:
            return batch
        

class CNNDataset(torch.utils.data.Dataset):
    def __init__(self, data, data_rcc, patient_ids, seg=None):
        self.data = data
        self.data_rcc = data_rcc
        self.patient_ids = patient_ids
        self.seg = seg

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        img = self.data[idx]
        label = self.data_rcc[idx]
        patient_id = self.patient_ids[idx]
        
        if self.seg:
            return  {'Patient_ID': patient_id, 'img': img, 'seg': self.seg[idx], 'label': label}

        else:
            return {'Patient_ID': patient_id, 'img': img, 'label': label}

   
