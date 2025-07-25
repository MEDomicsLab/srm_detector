"""
    @file:              utils.py
    @Author:            Moustafa Amine Bezzahi, Ihssene Brahimi 

    @Creation Date:     06/2024
    @Last modification: 11/2024

    @Description:       This file is used to define the Helper functions used in the pipeline.
"""

import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import nrrd


import monai
from monai.data import pad_list_data_collate

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import imageio
import csv
from scipy.stats import iqr, scoreatpercentile
from collections import Counter

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data.dataloader import default_collate

from sklearn.utils import resample

from constants import *
from data.dataset import SRMDataset, CNNDataset, ROIDataset
from data.roi import bounding_box, crop
from data.transforms import get_srm_transforms, InfiniteSampler

#################################################################################
#                              Helper Functions for Features                    #
#################################################################################



def compute_distribution(y_list):
    counter = Counter(y_list)
    total_count = sum(counter.values())
    distribution = {label: count / total_count for label, count in counter.items()}
    return distribution


def extract_features(image):
    
    median = np.median(image)
    energy = np.sum(np.power(image, 2))
    image_iqr = iqr(image)
    mean = np.mean(image)
    std = np.std(image)
    min = np.min(image)
    max = np.max(image)
    range = np.ptp(image)    
    per_10= scoreatpercentile(image, 10)  # 10th percentile
    per_90= scoreatpercentile(image, 90)  # 90th percentile
    qcd = iqr(image)/(scoreatpercentile(image, 75) + scoreatpercentile(image, 25))
    return [median, energy, image_iqr, mean, std, min, max, range, per_10, per_90, qcd]


def stack_deep_radiomics_features(data_loader, intermediate_features, reduce_radiomics_to=3):
    
    """
    
    This function stacks deep radiomics features extracted from the images along with intermediate features 
    obtained from a deep learning model. The final output is a combined feature array for further analysis.

    
    Parameters:
    -------

    data_loader: DataLoader
        The DataLoader providing the dataset of patient images and labels.
    
    intermediate_features: torch.Tensor
        The intermediate features extracted from a CNN model for each image.
    
    reduce_radiomics_to: int, optional (default=3)
        The number of radiomic features to retain for each image.

    Returns:
    -------
    X_train: np.ndarray
        The combined feature array including radiomic and CNN-extracted features.
    """
    
    # Initialize lists to store features and related data
    features_lesion_list_pos = []
    images_list_pos = []
    cclabels_pos = []
    patients_pos = []
    pos_features = []
    median_list = []
    energy_list = []
    iqr_list = []
    pos_dict = dict()

    # Process each batch of data in the data_loader
    for pos_data in tqdm(data_loader, desc="Processing ccRCC Patients ..."):
        
        root, images, labels = pos_data['Patient_ID'], pos_data['img'], pos_data['label']
        
        # Extract features for each patient in the batch
        for patient, img, label in tqdm(zip(root, images, labels)):
            images_list_pos.append(img)
            hist_lesion_img = extract_features(img)
            features_lesion_list_pos.append(hist_lesion_img)
            cclabels_pos.append(label)
            patients_pos.append(patient)

    # Store the extracted data in a dictionary
    pos_dict["img"] = images_list_pos
    pos_dict["label"] = cclabels_pos
    pos_dict["radiomics"] = features_lesion_list_pos

    # Reduce radiomic features to the specified number
    for pos in pos_dict['radiomics']:
        pos_features.append(pos[:reduce_radiomics_to])

    # Separate out specific radiomic features
    for item in pos_features:
        median_list.append(item[0])
        energy_list.append(item[1])
        iqr_list.append(item[2])
    
    # Store the specific features in the dictionary
    pos_dict['median'] = median_list
    pos_dict['energy'] = energy_list
    pos_dict['image_iqr'] = iqr_list
    pos_dict['cnn_features'] = intermediate_features.cpu().numpy()

    # Combine all features into a single array
    X_train = np.hstack((
        np.array(pos_dict['median']).reshape(-1, 1),
        np.array(pos_dict['energy']).reshape(-1, 1),
        np.array(pos_dict['image_iqr']).reshape(-1, 1),
        pos_dict['cnn_features']
    ))

    return X_train



def get_intermediate_cnn_features(X_train_cnn, 
                              y_train_cnn,
                              id_train_cnn,
                              X_test_cnn,
                              y_test_cnn,
                              id_test_cnn,
                              aug_pos_data_dict,
                              aug_neg_data_dict,
                              train_index, 
                              val_index, 
                              cnn_model, 
                              train_bs=32,
                              val_bs=8,
                              test_bs=4,
                              device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    
    intermediate_outputs = []
    # Get same training and validation sets from cnn training
    X_train_fold = [X_train_cnn[i] for i in train_index[:-2]]
    y_train_fold = [y_train_cnn[i] for i in train_index[:-2]]
    id_train_fold = [id_train_cnn[i] for i in train_index[:-2]]

    X_val_fold = [X_train_cnn[i] for i in val_index]
    y_val_fold = [y_train_cnn[i] for i in val_index]
    id_val_fold = [id_train_cnn[i] for i in val_index]

    # Add augmented positive samples to the training set
    X_train_fold.extend(aug_pos_data_dict['img'])
    y_train_fold.extend(aug_pos_data_dict['label'])
    id_train_fold.extend(aug_pos_data_dict['Patient_ID'])

    # Add augmented negative samples to the training set (each sample 3 times)
    X_train_fold.extend(aug_neg_data_dict['img'])
    y_train_fold.extend(aug_neg_data_dict['label'])
    id_train_fold.extend(aug_neg_data_dict['Patient_ID'])
    # Create datasets for this fold
    train_fold_dataset = CNNDataset(data=X_train_fold, data_rcc=y_train_fold, patient_ids=id_train_fold)
    val_fold_dataset = CNNDataset(data=X_val_fold, data_rcc=y_val_fold, patient_ids=id_val_fold)
    test_fold_dataset = CNNDataset(data=X_test_cnn, data_rcc=y_test_cnn, patient_ids=id_test_cnn)

    # Create DataLoaders for this fold
    train_loader = FilteredDataLoader(train_fold_dataset, batch_size=train_bs, shuffle=True)
    val_loader = FilteredDataLoader(val_fold_dataset, batch_size=val_bs, shuffle=True)
    test_loader = DataLoader(test_fold_dataset, batch_size=test_bs, shuffle=False)
    
    # Collect intermediate outputs
    cnn_model.eval()  # Set the model to evaluation mode

    with torch.no_grad():

        for data in train_loader:
            inputs = data['img'].to(device)
            _, train_intermediate_output = cnn_model(inputs)
            train_intermediate_outputs.append(train_intermediate_output)

    # Concatenate all intermediate outputs
    train_intermediate_outputs = torch.cat(train_intermediate_outputs, dim=0)

    with torch.no_grad():

        for data in test_loader:
            inputs = data['img'].to(device)
            _, test_intermediate_output = cnn_model(inputs)
            test_intermediate_output.append(test_intermediate_output)

    # Concatenate all intermediate outputs batches
    test_intermediate_outputs = torch.cat(test_intermediate_output, dim=0)

    return train_intermediate_outputs, test_intermediate_outputs, train_loader, test_loader, y_train_fold



#################################################################################
#                            Helper Functions for Data Loading                  #
#################################################################################
def prepare_srm_dataset(keys = ("img", "seg"),
                        train_frac = 0.8, 
                        val_frac = 0.2):
    """
        Prepare Cropped Dataset using ROIDataset Class and save it in a separate folder
    """

    # Concatenate the dataframes
    unlabeled_train = pd.DataFrame(UNLABELED_TRAIN_DATA)

    # sort dataset
    tp = ['Renal-CHUS-0010', 'Renal-CHUS-0030', 'Renal-CHUS-0055', 'Renal-CHUS-0070', 'Renal-CHUS-0085', 'Renal-CHUS-0092', 'Renal-CHUS-0104', 'Renal-CHUS-0066', 'Renal-CHUS-0091'] # 66, 91, etc have one kidney
    unlabeled_train =  unlabeled_train[~unlabeled_train['root'].isin(tp)] #final_data[final_data['root'] not in tp] # patients 30, 104, 10, 85, 25, 63, 77
    unlabeled_train = unlabeled_train.sort_values(by='root')
    unlabeled_train.reset_index(drop=True, inplace=True)

    # Concatenate the dataframes
    labeled_train = pd.DataFrame(LABELED_TRAIN_DATA)

    # sort dataset
    labeled_train =  labeled_train[~labeled_train['root'].isin(tp)] #final_data[final_data['root'] not in tp] # patients 30, 104, 10, 85, 25, 63, 77
    labeled_train = labeled_train.sort_values(by='root')
    labeled_train.reset_index(drop=True, inplace=True)

    # Concatenate the dataframes
    unlabeled_test = pd.DataFrame(UNLABELED_TEST_DATA)

    # sort dataset
    unlabeled_test =  unlabeled_test[~unlabeled_test['root'].isin(tp)] #final_data[final_data['root'] not in tp] # patients 30, 104, 10, 85, 25, 63, 77
    unlabeled_test = unlabeled_test.sort_values(by='root')
    unlabeled_test.reset_index(drop=True, inplace=True)

    # Concatenate the dataframes
    labeled_test = pd.DataFrame(LABELED_TEST_DATA)

    # sort dataset
    labeled_test =  labeled_test[~labeled_test['root'].isin(tp)] #final_data[final_data['root'] not in tp] # patients 30, 104, 10, 85, 25, 63, 77
    labeled_test = labeled_test.sort_values(by='root')
    labeled_test.reset_index(drop=True, inplace=True)    


    # dataloader - for unlabeled cases
    n_train = int(train_frac * len(labeled_train['root'])) + 1
    n_val = min(len(labeled_train['root']) - n_train, int(val_frac * len(labeled_train['root'])))
    print(f"split: labeled train {n_train} val {n_val}, folder: {LABELED_DATA_PATH}")

    train_files = [{keys[0]: img, keys[1]: seg} for img, seg in zip(labeled_train['img'][:n_train], labeled_train['seg'][:n_train])]
    val_files = [{keys[0]: img, keys[1]: seg} for img, seg in zip(labeled_train['img'][-n_val:], labeled_train['seg'][-n_val:])]

    # create a training data loader for labeled data

    print(f"Loading Labeled train set - {len(train_files)} patients\n")
    labeled_train_transforms_img = get_srm_transforms("labeled_train", key=(keys[0]))
    labeled_train_transforms_seg = get_srm_transforms("labeled_train", key=(keys[1]))
    train_ds = ROIDataset(labeled_train, transform=[labeled_train_transforms_img, labeled_train_transforms_seg])
    labeled_sampler = InfiniteSampler(list_indices=list(range(len(train_files))), batch_size=16)
    labeled_train_loader = monai.data.DataLoader(
        train_ds,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        batch_sampler=labeled_sampler
    )

    # create a validation data loader for labeled data

    print(f"Loading Labeled validation set - {len(val_files)} patients\n")
    val_transforms_img = get_srm_transforms("val", key=(keys[0]))
    val_transforms_seg = get_srm_transforms("val", key=(keys[1]))

    val_ds = ROIDataset(labeled_test, transform=[val_transforms_img, val_transforms_seg])

    labeled_val_loader = monai.data.DataLoader(
        val_ds,
        batch_size=4,  # image-level batch to the sliding window method, not the window-level batch
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        collate_fn=pad_list_data_collate,
    )

    
    # dataloader - for unlabeled cases
    n_train_unlabeled = int(train_frac * len(unlabeled_train['root'])) + 1
    n_val_unlabeled = min(len(unlabeled_train['root']) - n_train_unlabeled, int(val_frac * len(unlabeled_train['root'])))
    print(f"split: labeled train {n_train_unlabeled} val {n_val_unlabeled}, folder: {UNLABELED_DATA_PATH}")

    unlabeled_train_files = [{keys[0]: img, keys[1]: seg} for img, seg in zip(unlabeled_train['img'][:n_train_unlabeled], unlabeled_train['seg'][:n_train_unlabeled])]
    unlabeled_val_files = [{keys[0]: img, keys[1]: seg} for img, seg in zip(unlabeled_train['img'][-n_val_unlabeled:], unlabeled_train['seg'][-n_val_unlabeled:])]


    # create a training data loader for unlabeled data
    print(f"Loading Unlabeled train set - {len(unlabeled_train_files)} patients\n")
    unlabeled_train_transforms_img = get_srm_transforms("unlabeled_train", key=(keys[0]))
    unlabeled_train_transforms_seg = get_srm_transforms("unlabeled_train", key=(keys[1]))

    unlabeled_train_ds = ROIDataset(unlabeled_train, transform=[unlabeled_train_transforms_img, unlabeled_train_transforms_seg])

    unlabeled_train_loader = monai.data.DataLoader(
        unlabeled_train_ds,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        collate_fn=pad_list_data_collate,
    )

    # create a validation data loader for unlabeled data
    print(f"Loading Unlabeled train set - {len(unlabeled_val_files)} patients\n")

    val_unlabeled_transforms_img = get_srm_transforms(mode="val", key=(keys[0]))
    val_unlabeled_transforms_seg = get_srm_transforms(mode="val", key=(keys[1]))

    val_unlabeled_ds = ROIDataset(unlabeled_test, transform=[val_unlabeled_transforms_img, val_unlabeled_transforms_seg])
    unlabeled_val_loader = monai.data.DataLoader(
        val_unlabeled_ds,
        batch_size=4,  # image-level batch to the sliding window method, not the window-level batch
        num_workers=4,
        collate_fn=pad_list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )


    return labeled_train_loader, labeled_val_loader, unlabeled_train_loader, unlabeled_val_loader, labeled_test, unlabeled_test

def load_srm_dataset(predictions_dir):
    cropped_data = {
        "Patient_ID": [],
        "img": [],
        "seg": []
    }
    
    # List all patient directories
    patient_dirs = [d for d in os.listdir(predictions_dir) if os.path.isdir(os.path.join(predictions_dir, d))]
    
    for patient_id in patient_dirs:
        patient_dir = os.path.join(predictions_dir, patient_id)
        
        # Define file paths for image and segmentation
        image_filename = os.path.join(patient_dir, f'cropped_image_{patient_id}.nrrd')
        seg_filename = os.path.join(patient_dir, f'cropped_image_{patient_id}.seg.nrrd')
        
        # Read the image and segmentation files
        image, _ = nrrd.read(image_filename)
        seg, _ = nrrd.read(seg_filename)
        
        # Append data to the lists in the dictionary
        cropped_data["Patient_ID"].append(patient_id)
        cropped_data["img"].append(image)
        cropped_data["seg"].append(seg)
    
    return cropped_data


def prepare_test_data(test_data_path):
    # Concatenate the dataframes
    data = pd.concat([test_data_path], ignore_index=True)
    # sort dataset
    data = data.sort_values(by='root')

   

def prepare_classification_data(data_name):
            
    """
        Reads and Organize Clinical Data for three different classification tasks.
        
        Note 
        ---

        Following patients (having one kidney) were excluded in this function for all tasks : ['Renal-CHUS-0010', 'Renal-CHUS-0030', 'Renal-CHUS-0055', 'Renal-CHUS-0070', 'Renal-CHUS-0085', 'Renal-CHUS-0092', 'Renal-CHUS-0104', 'Renal-CHUS-0066', 'Renal-CHUS-0091'] 
        
        Parameters
        ---
        data_name: To select the classification task name: "ccRCC_vs_non_ccRCC" or "grade" or "subtype"
    """

    df1 = pd.read_csv('/projects/renal/DATA_CSV/Données_Clinique__CSV.csv')
    df1_selected = df1.iloc[:, [0, 7]]

    if data_name == "ccRCC_vs_non_ccRCC":

        df2 = pd.read_csv('/projects/renal/DATA_CSV/Outcome_2.csv')

        df2_selected = df2.iloc[:, [0, 1]]
        df1_selected.iloc[:, 1] = df1_selected.iloc[:, 1].map({'Right': 1 , 'Left': 0}) # Right is [1,0], Left is [0,1]

        df1_selected = df1_selected.rename(columns={'Patient_ID': 'root'})
        data_rcc = pd.read_csv("/projects/renal/DATA_CSV/Outcome_2.csv")


        print('Concatenation ...')
        # Concatenate the dataframes
        data = pd.concat([UNLABELED_TRAIN_DATA, UNLABELED_TEST_DATA, LABELED_TRAIN_DATA, LABELED_TEST_DATA], ignore_index=True)
        # sort dataset
        data = data.sort_values(by='root')

        clinical_data = df1_selected.rename(columns={'LESION_SIDE_IMAGING': 'tumor_location'}) # take tumor presence from original clinical data


        # Keep only patients who exist in image dataset and rcc dataset
        filtered_data_rcc = data_rcc[data_rcc['Patient_ID'].isin(data['root'])]
        filtered_data = data[data['root'].isin(filtered_data_rcc['Patient_ID'])]

        filtered_data.reset_index(drop=True, inplace=True)
        filtered_data_rcc.reset_index(drop=True, inplace=True)

        print(f"Number of Patients ready for training: {len(filtered_data)}")

        clinical_data = clinical_data[clinical_data['root'].isin(filtered_data['root'])]
        final_data = filtered_data[filtered_data['root'].isin(clinical_data['root'])]

        final_data = final_data.merge(clinical_data)
        final_data = final_data[final_data['root'] != 'Renal-CHUS-0055']  # problematic case
        final_data.reset_index(drop=True, inplace=True)

        filtered_data_rcc = filtered_data_rcc[filtered_data_rcc['Patient_ID'] != 'Renal-CHUS-0055']  
        filtered_data_rcc.reset_index(drop=True, inplace=True)


        tp = ['Renal-CHUS-0010', 'Renal-CHUS-0030', 'Renal-CHUS-0055', 'Renal-CHUS-0070', 'Renal-CHUS-0085', 'Renal-CHUS-0092', 'Renal-CHUS-0104', 'Renal-CHUS-0066', 'Renal-CHUS-0091'] # 66, 91, etc have one kidney
        final_data =  final_data[~final_data['root'].isin(tp)] #final_data[final_data['root'] not in tp] # patients 30, 104, 10, 85, 25, 63, 77
        filtered_data_rcc = filtered_data_rcc[~filtered_data_rcc['Patient_ID'].isin(tp)]
        final_data = final_data.reset_index(drop=True)
        filtered_data_rcc = filtered_data_rcc.reset_index(drop=True)

        data , target = final_data, filtered_data_rcc['PATHOLOGY_ccRCC_VS_non-ccRCC']
        return data, target
    
    elif data_name == "grade":


        df1_selected.iloc[:, 1] = df1_selected.iloc[:, 1].map({'Right': 1 , 'Left': 0}) # Right is [1,0], Left is [0,1]

        df1_selected = df1_selected.rename(columns={'Patient_ID': 'root'})

        df_grade = pd.read_csv('/projects/renal/DATA_CSV/Outcome_4.csv')
        df_grade = df_grade.iloc[:, [0, 1]]

        print('Concatenation ...')
        # Concatenate the dataframes
        data = pd.concat([UNLABELED_TRAIN_DATA, UNLABELED_TEST_DATA, LABELED_TRAIN_DATA, LABELED_TEST_DATA], ignore_index=True)
        # sort dataset
        data = data.sort_values(by='root')

        clinical_data = df1_selected.rename(columns={'LESION_SIDE_IMAGING': 'tumor_location'}) # take tumor presence from original clinical data


        # Keep only patients who exist in image dataset and rcc dataset
        filtered_data_grade = df_grade[df_grade['Patient_ID'].isin(data['root'])]
        filtered_data = data[data['root'].isin(filtered_data_grade['Patient_ID'])]

        filtered_data.reset_index(drop=True, inplace=True)
        filtered_data_grade.reset_index(drop=True, inplace=True)

        print(f"Number of Patients ready for training: {len(filtered_data)}")

        clinical_data = clinical_data[clinical_data['root'].isin(filtered_data['root'])]
        final_data = filtered_data[filtered_data['root'].isin(clinical_data['root'])]

        final_data = final_data.merge(clinical_data)
        final_data = final_data[final_data['root'] != 'Renal-CHUS-0055']  # problematic case
        final_data.reset_index(drop=True, inplace=True)

        filtered_data_grade = filtered_data_grade[filtered_data_grade['Patient_ID'] != 'Renal-CHUS-0055']  
        filtered_data_grade.reset_index(drop=True, inplace=True)


        tp = ['Renal-CHUS-0010', 'Renal-CHUS-0030', 'Renal-CHUS-0055', 'Renal-CHUS-0070', 'Renal-CHUS-0085', 'Renal-CHUS-0092', 'Renal-CHUS-0104', 'Renal-CHUS-0066', 'Renal-CHUS-0091']
        final_data =  final_data[~final_data['root'].isin(tp)] #final_data[final_data['root'] not in tp] # patients 30, 104, 10, 85, 25, 63, 77
        filtered_data_grade = filtered_data_grade[~filtered_data_grade['Patient_ID'].isin(tp)]
        final_data = final_data.reset_index(drop=True)
        filtered_data_grade = filtered_data_grade.reset_index(drop=True)

        data , target = final_data, filtered_data_grade['PATHOLOGY_GRADE']
        
        return data, target

    elif data_name == "subtype":

        df1_selected.iloc[:, 1] = df1_selected.iloc[:, 1].map({'Right': 1 , 'Left': 0}) # Right is [1,0], Left is [0,1]

        df1_selected = df1_selected.rename(columns={'Patient_ID': 'root'})

        df_grade = pd.read_csv('/projects/renal/DATA_CSV/Outcome_3.csv')
        df_grade = df_grade.iloc[:, [0, 1]]

        print('Concatenation ...')
        # Concatenate the dataframes
        data = pd.concat([UNLABELED_TRAIN_DATA, UNLABELED_TEST_DATA, LABELED_TRAIN_DATA, LABELED_TEST_DATA], ignore_index=True)
        # sort dataset
        data = data.sort_values(by='root')

        clinical_data = df1_selected.rename(columns={'LESION_SIDE_IMAGING': 'tumor_location'}) # take tumor presence from original clinical data


        # Keep only patients who exist in image dataset and rcc dataset
        filtered_data_grade = df_grade[df_grade['Patient_ID'].isin(data['root'])]
        filtered_data = data[data['root'].isin(filtered_data_grade['Patient_ID'])]

        filtered_data.reset_index(drop=True, inplace=True)
        filtered_data_grade.reset_index(drop=True, inplace=True)

        print(f"Number of Patients ready for training: {len(filtered_data)}")

        clinical_data = clinical_data[clinical_data['root'].isin(filtered_data['root'])]
        final_data = filtered_data[filtered_data['root'].isin(clinical_data['root'])]

        final_data = final_data.merge(clinical_data)
        final_data = final_data[final_data['root'] != 'Renal-CHUS-0055']  # problematic case
        final_data.reset_index(drop=True, inplace=True)

        filtered_data_grade = filtered_data_grade[filtered_data_grade['Patient_ID'] != 'Renal-CHUS-0055']  
        filtered_data_grade.reset_index(drop=True, inplace=True)


        tp = ['Renal-CHUS-0010', 'Renal-CHUS-0030', 'Renal-CHUS-0055', 'Renal-CHUS-0070', 'Renal-CHUS-0085', 'Renal-CHUS-0092', 'Renal-CHUS-0104', 'Renal-CHUS-0066', 'Renal-CHUS-0091']
        final_data =  final_data[~final_data['root'].isin(tp)] #final_data[final_data['root'] not in tp] # patients 30, 104, 10, 85, 25, 63, 77
        filtered_data_grade = filtered_data_grade[~filtered_data_grade['Patient_ID'].isin(tp)]
        final_data = final_data.reset_index(drop=True)
        filtered_data_grade = filtered_data_grade.reset_index(drop=True)

        data , target = final_data, filtered_data_grade['PATHOLOGY_ccRCC_VS_pap_']
        
        return data, target

    else:
        assert("Enter a valid data name !!!")
    
    

def find_srm_files(root_dir):

        """Read Images and Masks from folders and return them as lists"""

        images = []
        segments = []

        # Traverse through all subdirectories and files in the root directory
        for roots, dirs, files in os.walk(root_dir):
            for dir_name in dirs:
                if dir_name == "CECT":
                    cect_dir = os.path.join(roots, dir_name)
                    # Check if the CECT directory contains the required files
                    image_file = os.path.join(cect_dir, "CECT.nrrd")
                    segment_file = os.path.join(cect_dir, "CECT_Segmentation.seg.nrrd")
                    if os.path.exists(image_file):
                        images.append(image_file)
                        if os.path.exists(segment_file):
                            segments.append(segment_file)
        return images, segments


def get_random_subset(aug_pos_data_dict, aug_neg_data_dict, num_pos, num_neg):
    pos_indices = np.random.choice(len(aug_pos_data_dict['img']), num_pos, replace=False)
    neg_indices = np.random.choice(len(aug_neg_data_dict['img']), num_neg, replace=False)

    # Extract subsets using list comprehensions
    aug_pos_subset = {key: [aug_pos_data_dict[key][i] for i in pos_indices] for key in aug_pos_data_dict}
    aug_neg_subset = {key: [aug_neg_data_dict[key][i] for i in neg_indices] for key in aug_neg_data_dict}
    
    return aug_pos_subset, aug_neg_subset


def find_kidney_files(root_dir, labeled=False):  # labeled and unlabaled tags refers to kidney GT availability
    """Read Images and Masks from folders and return them as lists"""
    if labeled == True:  # there is kidney GT
    
        images = []
        roi_gt = []

        # Traverse through all subdirectories and files in the root directory
        for root, dirs, files in os.walk(root_dir):
            for dir_name in dirs:
                if dir_name == "CECT":
                    cect_dir = os.path.join(root, dir_name)
                    # Check if the CECT directory contains the required files
                    image_file = os.path.join(cect_dir, "CECT.nrrd")
                    segment_file = os.path.join(cect_dir, "CECT_Kidneys_Segmentation.seg.nrrd")
                    if os.path.exists(image_file):
                        images.append(image_file)
                        if os.path.exists(segment_file):  # if the manual kidney segmentation exists take it
                            roi_gt.append(segment_file)
                        else :  # else take the original tumor segmentation
                            segment_file = os.path.join(cect_dir, "CECT_Segmentation.seg.nrrd")
                            roi_gt.append(segment_file)
        return images, roi_gt  # segmentation provided is for kidney
    else :  # No kidney GT (We need MT predictions)

        images = []
        srm_gt = []  # if tumor GT is available
        pseudo_roi = []  # if kidney pseudo label (MT prediction) is available

        # Traverse through all subdirectories and files in the root directory
        for roots, dirs, files in os.walk(root_dir):
            for dir_name in dirs:
                if dir_name == "CECT":
                    cect_dir = os.path.join(roots, dir_name)
                    # Check if the CECT directory contains the required files
                    image_file = os.path.join(cect_dir, "CECT.nrrd")
                    srm_gt_file = os.path.join(cect_dir, "CECT_Segmentation.seg.nrrd")
                    pseudo_roi_file = os.path.join(roots, "mt_prediction.seg.nrrd") 
                    if os.path.exists(image_file):
                        images.append(image_file)
                        if os.path.exists(srm_gt_file):
                            srm_gt.append(srm_gt_file)
                        if os.path.exists(pseudo_roi_file):
                            pseudo_roi.append(pseudo_roi_file)
        return images, srm_gt, pseudo_roi   # segmentation provided is for tumor and not kidney, which is not util at this level



def custom_collate(batch):
    imgs = [item['img'] for item in batch]
    labels = [item['label'] for item in batch]
    ids = [item['Patient_ID'] for item in batch]
    return {'img': default_collate(imgs), 'label': default_collate(labels), 'Patient_ID': default_collate(ids)}


def get_test_data(img, gt):
    return SRMDataset(data=img, data_rcc=gt) 


def pos_neg_aug_datasets(data, target, aug_pos_rate=None, aug_neg_rate=None):

    # Separate data based on target labels
    negative_indices = target[target == 0] 
    positive_indices = target[target == 1]

    # Separate data based on negative_indices and positive_indices
    negative_data = {key: [data[key][i] for i in negative_indices.index] for key in data.keys()}
    positive_data = {key: [data[key][i] for i in positive_indices.index] for key in data.keys()}

    print("Negative Data:")
    print(len(negative_data['root']))

    print("\nPositive Data:")
    print(len(positive_data['root']))

    negative_indices = negative_indices.reset_index(drop=True)
    positive_indices = positive_indices.reset_index(drop=True)

    # Create datasets
    positive_dataset = SRMDataset(data=positive_data, data_rcc=positive_indices, augment_pos=True, aug_pos_rate=aug_pos_rate) 
    negative_dataset = SRMDataset(data=negative_data, data_rcc=negative_indices, augment_neg=True, aug_neg_rate=aug_neg_rate) 

    # augment the minority class (non ccRCC)

    # Assuming negative_dataset is a list of tuples (batch, aug_batch)
    neg_dataset = []
    aug_neg_dataset = []
    pos_dataset = []
    aug_pos_dataset = []

    # Iterate over the negative dataset
    for e in tqdm(negative_dataset):
        batch, aug_batch = e
        
        # Append original samples to neg_dataset
        neg_dataset.append(batch)
        
        # Check if aug_batch contains multiple samples
        if isinstance(aug_batch['img'], list):
            # If aug_batch contains a list of images, extend the list
            for img, seg in zip(aug_batch['img'], aug_batch['seg']):
                aug_neg_dataset.append({
                    'Patient_ID': aug_batch['Patient_ID'],
                    'img': img,
                    'seg': seg,
                    'label': aug_batch['label']
                })
        else:
            # If aug_batch contains a single image, append directly
            aug_neg_dataset.append(aug_batch)

    

    for e in tqdm(positive_dataset):
        batch, aug_batch = e
        
        # Append original samples to neg_dataset
        pos_dataset.append(batch)
        
        # Check if aug_batch contains multiple samples
        if isinstance(aug_batch['img'], list):
            # If aug_batch contains a list of images, extend the list
            for img, seg in zip(aug_batch['img'], aug_batch['seg']):
                aug_pos_dataset.append({
                    'Patient_ID': aug_batch['Patient_ID'],
                    'img': img,
                    'seg': seg,
                    'label': aug_batch['label']
                })
        else:
            # If aug_batch contains a single image, append directly
            aug_pos_dataset.append(aug_batch)


    # Check the length of the datasets
    print(f"Total original negative samples: {len(neg_dataset)}")
    print(f"Total augmented negative samples: {len(aug_neg_dataset)}")
    print(f"Total original positive samples: {len(pos_dataset)}")
    print(f"Total augmented positive samples: {len(aug_pos_dataset)}")

    return pos_dataset, aug_pos_dataset, neg_dataset, aug_neg_dataset

def get_clf_data_dict(data_loader):

    """
    Returns a dictionary with image tensor, patiend ID, label and radiomics

    """

    features_lesion_list = []
    images_list = []
    cclabels = []
    patients = []

    for data in tqdm(data_loader, desc="Processing Patients ..."):
        
        root, images, labels= data['Patient_ID'], data['img'], data['label']
        for patient, img, label in tqdm(zip(root, images, labels)):
            images_list.append(img)
            hist_lesion_img = extract_features(img)
            features_lesion_list.append(hist_lesion_img)
            cclabels.append(label)
            patients.append(patient)
    
    data_dict = dict()
    data_dict["img"] = images_list
    data_dict["Patient_ID"] = patients
    data_dict["label"] = cclabels
    data_dict["radiomics"] = features_lesion_list

    return data_dict



# Function to filter out images with too much background
def filter_background_images(images, labels, patients, threshold = 0.3):
    filtered_images = []
    filtered_labels = []
    filtered_patients = []
    for i in range(len(images)):
        image = images[i]
        label = labels[i]
        id = patients[i]

        # Calculate the percentage of background pixels

        background_mask = torch.abs(image) < threshold
        background_pixels = torch.sum(background_mask).item()  # Sum of background pixels

        #background_pixels = torch.sum(image == background_value)
        total_pixels = image.numel()
        background_percentage = background_pixels / total_pixels
        #print("background % : ", background_percentage)

        # Include the image if less than 50% of the pixels are background
        if background_percentage < threshold:
        #print(np.unique(image.cpu().numpy()))
            filtered_images.append(image)
            filtered_labels.append(label)
            filtered_patients.append(id)

    return filtered_images, filtered_labels, filtered_patients



# Function to maintain batch size
def maintain_batch_size(filtered_images, filtered_labels, filtered_patients, dataset, batch_size):
    additional_images = []
    additional_labels = []
    additional_patients = []

    dataset_iter = iter(DataLoader(dataset, batch_size=1, shuffle=True))

    while len(filtered_images) < batch_size:
        try:
            data = next(dataset_iter)
            image = data['img'].squeeze(0)
            label = data['label'].squeeze(0)
            id = data['Patient_ID']
            # Calculate the percentage of background pixels
            #background_pixels = torch.sum(image == 0)
            background_mask = torch.abs(image) < 0.3
            background_pixels = torch.sum(background_mask).item()  # Sum of background pixels
            total_pixels = image.numel()
            background_percentage = background_pixels / total_pixels

            # Include the image if less than 50% of the pixels are background
            if background_percentage < 0.3:
              additional_images.append(image)
              additional_labels.append(label)
              additional_patients.append(id)
        except StopIteration:
            break

    # Append additional images and labels to filtered ones
    filtered_images.extend(additional_images)
    filtered_labels.extend(additional_labels)
    filtered_patients.extend(additional_patients)

    # Ensure the batch size is correct
    filtered_images = filtered_images[:batch_size]
    filtered_labels = filtered_labels[:batch_size]
    filtered_patients = filtered_patients[:batch_size]

    # Convert lists back to tensors
    filtered_batch = {
        'img': torch.stack(filtered_images),
        'label': torch.stack(filtered_labels),
        'Patient_ID': filtered_patients
    }

    return filtered_batch

# Custom DataLoader class
class FilteredDataLoader(DataLoader):
    def __iter__(self):
        for batch in super().__iter__():
            images = batch['img']
            labels = batch['label']
            patients = batch['Patient_ID']
            filtered_images, filtered_labels, filtered_patients = filter_background_images(images, labels, patients)
            filtered_batch = maintain_batch_size(filtered_images, filtered_labels, filtered_patients, self.dataset, batch_size=len(images))
            yield filtered_batch


#################################################################################
#                         Helper Functions for Visualization                    #
#################################################################################


def visualize_batch(batch, image_only=True, style='viridis'):
    images = batch['img']
    labels = batch['label']
    #print(labels)
    ids = batch['Patient_ID'][0]
    batch_size = len(images)
    # Assuming batch size is 8
    #num_images = images.size(0)
    fig, axes = plt.subplots(1, batch_size, figsize=(15, 5))
    
    for i in range(batch_size):
        image = images[i].numpy()  # Convert to numpy array
        label = labels[i].item()   # Convert to scalar
        id = ids[i]
        # Select a slice from the 3D volume (e.g., middle slice)
        slice_index = image.shape[4] // 2
        axes[i].imshow(image[0, 0, :, :, slice_index], cmap=style)
        axes[i].set_title(f'patient: {id[-4:]}')
        axes[i].axis('off')

    plt.show()



def visualize_batch(batch, image_only = True, style='viridis'):
    images = batch['img']
    
    labels = batch['label']
    ids = batch['Patient_ID']
    batch_size = len(images)
        

    if image_only:

        fig, axes = plt.subplots(nrows=1, ncols=batch_size, figsize=(24, 16))  # or 24, 8
        for i in range(batch_size):
            
            image = images[i].numpy()
            label = labels[i].item()
            id = ids[i]
            
            if label == 0:
                label = f"{id[-4:]} nonRCC"
            else:
                label = f"{id[-4:]} RCC"

            # Plot the image in the first row
            axes[i].imshow(image[0,:,:,15], cmap=style) # viridis
            axes[i].set_title(label)
            axes[i].axis('off')

    else:

        masks = batch['seg']
        fig, axes = plt.subplots(nrows=2, ncols=batch_size, figsize=(24, 16))

        for i in range(batch_size):
            
            image = images[i].numpy()
            mask = masks[i].numpy()
            label = labels[i].item()
            id = ids[i]
            
            if label == 0:
                label = f"{id[-4:]} nonRCC"
            else:
                label = f"{id[-4:]} RCC"

            # Plot the image in the first row
            axes[0, i].imshow(image[0,:,:,15], cmap='gray')
            axes[0, i].set_title(label)
            axes[0, i].axis('off')

            # Plot the mask in the second row
            axes[1, i].imshow(mask[0,:,:,15], cmap='gray')
            axes[1, i].set_title(f"Mask {id[-4:]}")
            axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()


def plot_imbalance(train_dataset, test_dataset, dataset_name):
    ''' 
    Creates the subplots and plots the class distribution
    '''
    class_count_train = check_imbalance(train_dataset)
    class_count_test = check_imbalance(test_dataset)

    fig, axs = plt.subplots(1, 2, figsize = (24, 8))
    fig.suptitle(f'Class distribution for "{dataset_name}" dataset in Training and Test sets', size = 15)
    if dataset_name == "RCC_vs_non_ccRCC":
        labels = ['0: Non ccRCC', '1: ccRCC']
    elif dataset_name == "subtype":
        labels = ['0: papRCC', '1: ccRCC']
    elif dataset_name == "grade":
        labels = ['0: Low Grade', '1: High Grade']

    axs[0].bar(labels, class_count_train)
    axs[0].set_title('Train set')
    axs[0].set_xlabel('Class', size = 12)
    axs[0].set_ylabel('Number of samples')

    axs[1].bar(labels, class_count_test)
    axs[1].set_title('Test set')
    axs[1].set_xlabel('Class', size = 12)
    axs[1].set_ylabel('Number of samples')
    plt.show()



# Plotting train loss for all folds
def plot_metrics_folds(all_train_losses, all_train_accuracies, all_val_losses, all_val_accuracies, num_folds = 10):

    plt.figure(figsize=(12, 6))
    for fold in range(num_folds):
        plt.plot(all_train_losses[fold], label=f'Fold {fold+1}')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Across Folds')
    plt.legend()
    plt.show()

    # Plotting train accuracy for all folds
    plt.figure(figsize=(12, 6))
    for fold in range(num_folds):
        plt.plot(all_train_accuracies[fold], label=f'Fold {fold+1}')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy Across Folds')
    plt.legend()
    plt.show()

    # Plotting validation loss for all folds
    plt.figure(figsize=(12, 6))
    for fold in range(num_folds):
        plt.plot(all_val_losses[fold], label=f'Fold {fold+1}')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss Across Folds')
    plt.legend()
    plt.show()

    # Plotting validation accuracy for all folds
    plt.figure(figsize=(12, 6))
    for fold in range(num_folds):
        plt.plot(all_val_accuracies[fold], label=f'Fold {fold+1}')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy Across Folds')
    plt.legend()
    plt.show()



def plot_bounding_boxes(image, mask, pred_mask_sample, bounding_boxes, slice_num, patient_id):
    fig, axs = plt.subplots(1, 3, figsize=(12, 6))
    
    fig.suptitle(f"Patient ID: {patient_id}", fontsize=16)

    axs[0].set_title("Image")
    axs[0].imshow(image[0, :, :, slice_num], cmap="gray")
    axs[0].axis('off')

    axs[1].set_title("Ground Truth Mask")
    axs[1].imshow(mask[0, :, :, slice_num], cmap="viridis")
    axs[1].axis('off')

    axs[2].set_title("Predicted Mask")
    axs[2].imshow(pred_mask_sample[0, :, :, slice_num], cmap="viridis")
    axs[2].axis('off')

    for box in bounding_boxes:
        x_start, y_start, z_start, x_length, y_length, z_length = box
        if x_start <= slice_num < x_start + x_length:
            z_relative = slice_num - z_start
            rect = plt.Rectangle((y_start, z_start), y_length, z_length, edgecolor='r', facecolor='none', lw=2)
            axs[2].add_patch(rect)

    return fig



def plot_cropped_fig(crop_img, crop_seg, crop_pred, pred, crop_depth, box, patient_id):
    x_start = box[0]
    for slice_num in range(pred.shape[3]):
                
        # Plotting to visualize the rectangle on a slice if slice_num is within the ROI
        if x_start <= slice_num < x_start + crop_depth:  # if we are in the bbox zone of slices

                    fig, axs = plt.subplots(1, 3, figsize=(12, 6))
                                    
                    plt.suptitle(f"Patient ID {patient_id} - Slice {slice_num}")

                    x_relative = slice_num - x_start

                    axs[0].imshow(crop_img[0,:, :, x_relative], cmap='gray')
                    axs[0].set_title(f'Cropped Original Image')
                    axs[0].axis('off')


                    axs[1].imshow(crop_seg[0,:, :, x_relative], cmap='gray')
                    axs[1].set_title(f'Cropped Seg Image')
                    axs[1].axis('off')

                    cropped_slice = crop_pred[0,:, :, x_relative]
                    axs[2].imshow(cropped_slice, cmap='gray')
                    axs[2].set_title(f'Cropped Pred Image')
                    axs[2].axis('off')

                    plt.show()

def visualize_srm_batch(batch, image_only=True, style='viridis'):
    images = batch['img']
    ids = batch['Patient_ID']
    batch_size = len(images)

    if image_only:
        if batch_size == 1:
            fig, ax = plt.subplots(figsize=(24, 16))
            ax.imshow(images[0].numpy()[0, :, :, 15], cmap=style)
            ax.axis('off')
        else:
            fig, axes = plt.subplots(nrows=1, ncols=batch_size, figsize=(24, 16))
            for i in range(batch_size):
                image = images[i].numpy()
                axes[i].imshow(image[0, :, :, 15], cmap=style)
                axes[i].axis('off')
    else:
        masks = batch['seg']
        if batch_size == 1:
            fig, (ax_img, ax_mask) = plt.subplots(nrows=2, ncols=1, figsize=(24, 16))
            ax_img.imshow(images[0].numpy()[0, :, :, 15], cmap='gray')
            ax_img.axis('off')
            ax_mask.imshow(masks[0].numpy()[0, :, :, 15], cmap='gray')
            ax_mask.set_title(f"Mask {ids[0][-4:]}")
            ax_mask.axis('off')
        else:
            fig, axes = plt.subplots(nrows=2, ncols=batch_size, figsize=(24, 16))
            for i in range(batch_size):
                image = images[i].numpy()
                mask = masks[i].numpy()
                axes[0, i].imshow(image[0, :, :, 15], cmap='gray')
                axes[0, i].axis('off')
                axes[1, i].imshow(mask[0, :, :, 15], cmap='gray')
                axes[1, i].set_title(f"Mask {ids[i][-4:]}")
                axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()

def check_mask_values(batch):
    ids = batch['Patient_ID']
    masks = batch['seg']
    batch_size = len(masks)

    if batch_size == 1:
        mask = masks[0].numpy()
        unique_values = np.unique(mask)
        print(f"Unique label values for {ids[0]}: {unique_values}")

    else:
        for i in range(batch_size):
            mask = masks[i].numpy()
            
            unique_values = np.unique(mask)
            print(f"Unique label values for {ids[i]}: {unique_values}")


def plot_metrics(epoch_loss_values, metric_values, consistency_values, save_path, val_interval=2):
    epochs = list(range(1, len(epoch_loss_values) + 1))

    plt.figure(figsize=(14, 5))

    # Plot training loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, epoch_loss_values, label='Training Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)

    # Plot validation Dice score
    plt.subplot(1, 3, 2)
    plt.plot(range(val_interval, len(metric_values) * val_interval + 1, val_interval), metric_values, label='Validation Dice Score', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.title('Validation Dice Score')
    plt.legend()
    plt.grid(True)

    # Plot consistency loss
    plt.subplot(1, 3, 3)
    plt.plot(range(1, len(consistency_values) + 1), consistency_values, label='Consistency Loss', color='red')
    plt.xlabel('Iteration')
    plt.ylabel('Consistency Loss')
    plt.title('Consistency Loss')
    plt.legend()
    plt.grid(True)

    # Save the figure
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def overlay_segmentation_srm(vol, seg):
    # Scale volume to greyscale range
    vol_greyscale = (255*(vol - np.min(vol))/np.ptp(vol)).astype(int)
    # Convert volume to RGB
    vol_rgb = np.stack([vol_greyscale, vol_greyscale, vol_greyscale], axis=-1)
    # Initialize segmentation in RGB
    shp = seg.shape
    seg_rgb = np.zeros((shp[0], shp[1], shp[2], 3), dtype=int)  # Update dtype here
    # Set class to appropriate color
    seg_rgb[np.equal(seg, 1)] = [255, 0, 0]
    seg_rgb[np.equal(seg, 2)] = [0, 0, 255]
    # Get binary array for places where an ROI lives
    segbin = np.greater(seg, 0)
    repeated_segbin = np.stack((segbin, segbin, segbin), axis=-1)
    # Weighted sum where there's a value to overlay
    alpha = 0.3
    vol_overlayed = np.where(
        repeated_segbin,
        np.round(alpha*seg_rgb+(1-alpha)*vol_rgb).astype(np.uint8),
        np.round(vol_rgb).astype(np.uint8)
    )
    # Return final volume with segmentation overlay
    return vol_overlayed

def visualize_evaluation_srm(case_id, vol, truth, pred, eva_path):
    # Color volumes according to truth and pred segmentation
    vol_truth = overlay_segmentation_srm(vol, truth)
    vol_pred = overlay_segmentation_srm(vol, pred)
    # Create a figure and two axes objects from matplotlib
    fig, (ax1, ax2) = plt.subplots(1, 2)
    # Initialize the two subplots (axes) with an empty 512x512 image
    data = np.zeros(vol.shape[1:3])
    ax1.set_title("Ground Truth")
    ax2.set_title("Prediction")
    img1 = ax1.imshow(data)
    img2 = ax2.imshow(data)
    # Update function for both images to show the slice for the current frame
    def update(i):
        plt.suptitle("Case ID: " + str(case_id) + " - " + "Slice: " + str(i))
        img1.set_data(vol_truth[i])
        img2.set_data(vol_pred[i])
        return [img1, img2]
    # Compute the animation (gif)
    interval_ms = 100  # Adjust this value to change the speed (100 ms between frames)
    fps = 10  # Adjust this value to change the speed (frames per second)
    ani = animation.FuncAnimation(fig, update, frames=len(truth), interval=interval_ms,
                                  repeat_delay=0, blit=False)
    # Set up the output path for the gif
    if not os.path.exists(eva_path):
        os.mkdir(eva_path)
    file_name = "visualization.case_" + str(case_id).zfill(5) + ".gif"
    out_path = os.path.join(eva_path, file_name)
    # Save the animation (gif)
    ani.save(out_path, writer='imagemagick', fps=fps)
    # Close the matplotlib
    plt.close()

#################################################################################
#                               Helper Functions for Saving                     #
#################################################################################


def save_csv(data_dict, filename):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        for key, value in data_dict.items():
            writer.writerow([key, value])
             
def save_dict_as_csv(data, filename="bbox.csv"):
    # Get the keys (column names)
    columns = data.keys()

    # Open a new CSV file for writing
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write the column headers
        writer.writerow(columns)
        
        # Write the rows
        for row in zip(*data.values()):
            writer.writerow(row)

    return f"Dictionary saved as {filename}."


def save_gifs(data, post_proc, post_pred, output_folder=GIFS_FOLDER, tag=None):

    for i in range(len(data['root'])):  #len(data)
        # Generate a random integer i from 0 to 250
            
        img_sample = post_proc(data['img'][i])
        if "seg" in data.keys():
            mask_sample = post_pred(data['seg'][i])
        else:
            mask_sample = post_pred(data['pred'][i])

        # Calculate bounding boxes for ground truth mask and predicted mask
        pred_boxes = bounding_box(mask_sample)

        id = data['root'][i]
        # Generate and save the GIF
        if tag == 'labeled':
            gif_filename = f'labeled_bounding_boxes_{id}.gif'
        elif tag == 'unlabeled':
            gif_filename = f'unlabeled_bounding_boxes_{id}.gif'
        else:
            gif_filename = f'bounding_boxes_{id}.gif'
        
        with imageio.get_writer(os.path.join(output_folder,gif_filename), mode='I', duration=0.5) as writer:
            for idx in range(img_sample.size(3)):
                fig = plot_bounding_boxes(img_sample, mask_sample, mask_sample, pred_boxes, idx, data['root'][i])
                fig.canvas.draw()
                
                # Convert the figure to an image array
                image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
                image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                
                # Append the image to the GIF
                writer.append_data(image)
                
                # Close the figure to free memory
                plt.close(fig)

        print(f"GIF saved as {gif_filename}")


def save_crop_gif(resized_img, resized_seg, pred, box, patient_id, output_folder):    
    
        crop_img, crop_seg, crop_pred, crop_depth = crop(resized_img, resized_seg, pred, box)
        x_start = box[0]

        gif_filename = f'cropped_kidney_{patient_id}_{box}.gif'
        with imageio.get_writer(os.path.join(output_folder,gif_filename), mode='I', duration=0.5) as writer:

            for slice_num in range(pred.shape[3]):
                
                # Plotting to visualize the rectangle on a slice if slice_num is within the ROI
                if x_start <= slice_num < x_start + crop_depth:  # if we are in the bbox zone of slices
                    
                    fig, axs = plt.subplots(1, 3, figsize=(12, 6))
                    
                    plt.suptitle(f"Patient ID {patient_id} - Slice {slice_num}")

                    x_relative = slice_num - x_start

                    axs[0].imshow(crop_img[0,:, :, x_relative], cmap='gray')
                    axs[0].set_title(f'Cropped Original Image')
                    axs[0].axis('off')


                    axs[1].imshow(crop_seg[0,:, :, x_relative], cmap='gray')
                    axs[1].set_title(f'Cropped Seg Image')
                    axs[1].axis('off')

                    cropped_slice = crop_pred[0,:, :, x_relative]
                    axs[2].imshow(cropped_slice, cmap='gray')
                    axs[2].set_title(f'Cropped Pred Image')
                    axs[2].axis('off')
                    
                    #plt.show()

                    # save gif
                    fig.canvas.draw()
                    # Convert the figure to an image array
                    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
                    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    
                    # Append the image to the GIF
                    writer.append_data(image)

                    # Close the figure to free memory
                    plt.close(fig)

                else:
                    pass

        print(f"GIF saved as {gif_filename}")
    
def save_cropped_dataset(loader, predictions_dir):
    """
        Save cropped images and their masks in a special directory for further analysis
    """
    for batch in loader:
        image = batch["img"].cpu().numpy()
        seg = batch["seg"].cpu().numpy()
        patient_id = batch["Patient_ID"][0]

        # Create a directory for the current patient
        patient_dir = os.path.join(predictions_dir, str(patient_id))
        os.makedirs(patient_dir, exist_ok=True)

        # Define file paths for image and segmentation
        image_filename = os.path.join(patient_dir, f'cropped_image_{patient_id}.nrrd')
        seg_filename = os.path.join(patient_dir, f'cropped_image_{patient_id}.seg.nrrd')

        # Save the image and segmentation
        nrrd.write(image_filename, image)
        nrrd.write(seg_filename, seg)
        
        print(f'Patient {patient_id} data saved in {patient_dir}!')

#################################################################################
#                            Helper Functions for Mean Teacher                  #
#################################################################################

    
"""Functions for ramping hyperparameters up or down

Each function takes the current training step or epoch, and the
ramp length in the same format, and returns a multiplier between
0 and 1.
"""

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))


def get_current_consistency_weight(epoch, consistency=5, consistency_rampup=250.0):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return consistency * sigmoid_rampup(epoch, consistency_rampup)

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)  # student recent weights get more weight (ema params) while old params (previous teacher params) get less weight (1-alpha)




#################################################################################
#                               Helper Functions for Training                   #
#################################################################################


def get_sampler(dataset):
    '''  
    Description: Get weighted samples using Weighted Random Sampler
        Arguments:
            dataset: Any imbalanced dataset that requires oversampling
        Returns:
            dataset used with Weighted Random Sampler
    '''
    classes = [label['label'] for label in dataset]
    index_0 = [idx   for idx, label in enumerate(classes) if label == 0]
    index_1 = [idx   for idx, label in enumerate(classes) if label == 1]
    weights = torch.zeros(len(index_0) + len(index_1))
    weights[index_0] = 1.0 / len(index_0)
    weights[index_1] = 1.0 / len(index_1)
    sampler = WeightedRandomSampler(weights = weights, num_samples = len(weights), replacement = True)
    return sampler

def check_imbalance(dataset):
    ''' 
    Counts the number of samples in every class for a given dataset
    Arguments:
        dataset 
    Returns: 
        tuple (value for class 0, value for class 1)
    '''
    classes = [label['label']  for label in dataset]
    index_0 = len([idx   for idx, label in enumerate(classes) if label == 0])
    index_1 = len([idx   for idx, label in enumerate(classes) if label == 1])
    return index_0, index_1 


def dice_coef(y_true, y_pred, smooth=1.0):
    intersection = torch.sum(y_true * y_pred)
    dice = (2. * intersection + smooth) / (torch.sum(y_true) + torch.sum(y_pred) + smooth)
    return dice.item()

def iou(y_true, y_pred, num_classes):
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)
    iou = 0.0
    for cls in range(num_classes):
        true_class = y_true == cls
        pred_class = y_pred == cls
        intersection = torch.sum(true_class & pred_class).float()
        union = torch.sum(true_class | pred_class).float()
        if union == 0:
            iou += 1.0  # If there is no ground truth or prediction for this class
        else:
            iou += intersection / union
    return (iou / num_classes).item()