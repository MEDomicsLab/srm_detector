"""
    @file:              dataloaders.py
    @Author:            Ihssene Brahimi, Moustafa Amine Bezzahi

    @Creation Date:     05/2024
    @Last modification: 11/2024

    @Description:       This file is used to define both clinical and imaging data loaders.
"""

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import monai
from monai.data import pad_list_data_collate

from utils import *
from constants import *
from .transforms import get_kidney_transforms, get_srm_transforms, InfiniteSampler
from .dataset import ROIDataset

########## SEGMENTATION LOADER ##############
def load_segmentation_data(task_name,
                           labeled=None,  
                           roi=False, 
                           labeled_data_path=LABELED_DATA_PATH,
                           unlabeled_data_path=UNLABELED_DATA_PATH,
                           infer_data_path=None,
                           train_bs=16, 
                           val_bs=1, 
                           train_frac=0.8, 
                           val_frac=0.2,
                           inference_mode=False):

    """
       Load unlabeled and labeled datasets.
  
       Parameters
       ---
       task_name: load kidney data or tumour data. 
       labeled: tag could be "labeled" if we want to load the labeled data. The function returns the training set and validation set
       for images with available masks only.
       or "unlabeled" if we want to load the unlabeled data. The function returns the training set and validation set for images 
       without a mask.
       labeled_data_path: path for labeled set
       unlabeled_data_path: path for unlabeled set
       infer_data_path: path for inference set (if inference_mode is enabled)
       train_bs: Batch size for training set. 
       val_bs: Batch size for validation set.
       train_frac: Percentage of training set.
       val_frac: percentage of validation set.
       inference_mode: enable or disable inference mode.
       
    
    """

    if task_name=="kidney":

        keys = ("img", "seg", "pred")

        if inference_mode == True:
            
            # Find NRRD files for images and segments
            infer_images, roi_gt  = find_kidney_files(infer_data_path, labeled=True)  # because in inference we have the gt for roi (kidneys)

            # Print the list of images and segments
            print("Number of inference images:")
            print(len(infer_images))

            infer_images = sorted(infer_images)
            infer_patient_folders = [folder for folder in os.listdir(infer_data_path)
                                            if folder.startswith("Renal-CHUS-") and 
                                            os.path.exists(os.path.join(infer_data_path, folder, "CECT"))] 

            infer_patient_folders = sorted(infer_patient_folders)
            print(f"Inference: total images: ({len(infer_patient_folders)}) from folder: {infer_data_path}")

            # dataloader for test set
            infer_files = [{keys[0]: img, keys[1]: seg, "id": id} for img, seg, id in zip(infer_images, roi_gt, infer_patient_folders)]
            
            print(infer_files)
            
            # create a validation data loader for unlabeled data
            print(f"Loading inference set - {len(infer_files)} patients\n")
            infer_transforms = get_kidney_transforms(mode="val", keys=(keys[0], keys[1]))
            infer_ds = monai.data.CacheDataset(data=infer_files, transform=infer_transforms)
            infer_loader = monai.data.DataLoader(
                infer_ds,
                batch_size=1,  # image-level batch to the sliding window method, not the window-level batch
                num_workers=2,
                collate_fn=pad_list_data_collate,
                pin_memory=False #torch.cuda.is_available(),
            )

            return infer_loader
        
        if labeled == True:
            
            # Find NRRD files for images and segments
            labeled_images, labeled_labels = find_kidney_files(labeled_data_path, labeled=True)

            # Print the list of images and segments
            print("Number of labeled images:")
            print(len(labeled_images))

            print("\nNumber of labeled segment:")
            print(len(labeled_labels))

            # dataset - labeled data and unlabeled data
            labeled_images = sorted(labeled_images)
            labeled_labels = sorted(labeled_labels)
            labeled_patient_folders = [folder for folder in os.listdir(labeled_data_path)
                                            if folder.startswith("Renal-CHUS-") and 
                                            os.path.exists(os.path.join(labeled_data_path, folder, "CECT"))]
                                            

            print(f"training: total labeled image/label: ({len(labeled_patient_folders)}) from folder: {labeled_data_path}")
            
            # dataloader - for unlabeled cases
            n_train = int(train_frac * len(labeled_images)) + 1
            n_val = min(len(labeled_images) - n_train, int(val_frac * len(labeled_images)))
            print(f"split: labeled train {n_train} val {n_val}, folder: {labeled_data_path}")

            train_files = [{keys[0]: img, keys[1]: seg} for img, seg in zip(labeled_images[:n_train], labeled_labels[:n_train])]
            val_files = [{keys[0]: img, keys[1]: seg} for img, seg in zip(labeled_images[-n_val:], labeled_labels[-n_val:])]
            
            # create a training data loader for labeled data
        
            print(f"Loading Labeled train set - {len(train_files)} patients\n")
            labeled_train_transforms = get_kidney_transforms("labeled_train", keys=(keys[0], keys[1]))
            if roi == True:  # what is this used for??? 
                data = pd.concat([UNLABELED_TRAIN_DATA, UNLABELED_TEST_DATA, LABELED_TRAIN_DATA, LABELED_TEST_DATA], ignore_index=True)
                # sort dataset
                data = data.sort_values(by='root')
                data.reset_index(drop=True, inplace=True)
                train_ds = ROIDataset(data)
            else:
                train_ds = monai.data.CacheDataset(data=train_files, transform=labeled_train_transforms)
            labeled_sampler = InfiniteSampler(list_indices=list(range(len(train_files))), batch_size=train_bs)
            labeled_train_loader = monai.data.DataLoader(
                train_ds,
                num_workers=4,
                pin_memory=False,#torch.cuda.is_available(),
                batch_sampler=labeled_sampler
            )

            # create a validation data loader for labeled data

            print(f"Loading Labeled validation set - {len(val_files)} patients\n")
            val_transforms = get_kidney_transforms("val", keys=(keys[0], keys[1]))
            val_ds = monai.data.CacheDataset(data=val_files, transform=val_transforms)
            labeled_val_loader = monai.data.DataLoader(
                val_ds,
                batch_size=val_bs,  # image-level batch to the sliding window method, not the window-level batch
                num_workers=2,
                pin_memory=False, #torch.cuda.is_available(),
                collate_fn=pad_list_data_collate,
            )

            return labeled_train_loader, labeled_val_loader

        else:
        
            # Find NRRD files for images and segments
            unlabeled_images, srm_gt, pseudo_roi = find_kidney_files(unlabeled_data_path)  # 2nd element represents the gt for tumor which we ignore now

            # Print the list of images and segments
            print("Number of unlabeled images:")
            print(len(unlabeled_images))

            unlabeled_images = sorted(unlabeled_images)
            unlabeled_patient_folders = [folder for folder in os.listdir(unlabeled_data_path)
                                            if folder.startswith("Renal-CHUS-") and 
                                            os.path.exists(os.path.join(unlabeled_data_path, folder, "CECT"))] #[:150]

            print(f"training: total predicted unlabeled image: ({len(unlabeled_patient_folders)}) from folder: {unlabeled_data_path}")


            # dataloader - for unlabeled cases
            n_train_unlabeled = int(train_frac * len(unlabeled_images)) + 1
            n_val_unlabeled = min(len(unlabeled_images) - n_train_unlabeled, int(val_frac * len(unlabeled_images)))
            print(f"split: labeled train {n_train_unlabeled} val {n_val_unlabeled}, folder: {unlabeled_data_path}")

            unlabeled_train_files = [{keys[0]: img, keys[1]: seg, keys[2]: pred} for img, seg, pred in zip(unlabeled_images[:n_train_unlabeled], srm_gt[:n_train_unlabeled], pseudo_roi[:n_train_unlabeled])]
            unlabeled_val_files = [{keys[0]: img, keys[1]: seg, keys[2]: pred} for img, seg, pred in zip(unlabeled_images[-n_val_unlabeled:], srm_gt[-n_val_unlabeled:], pseudo_roi[-n_val_unlabeled:])]
            
            
            # create a training data loader for unlabeled data
            print(f"Loading Unlabeled train set - {len(unlabeled_train_files)} patients\n")
            unlabeled_train_transforms = get_kidney_transforms("unlabeled_train", keys)
            if roi == True:
                data = pd.concat([UNLABELED_TRAIN_DATA, UNLABELED_TEST_DATA, LABELED_TRAIN_DATA, LABELED_TEST_DATA], ignore_index=True)
                # sort dataset
                data = data.sort_values(by='root')
                data.reset_index(drop=True, inplace=True)
                unlabeled_train_ds = ROIDataset(data)
            else:
                unlabeled_train_ds = monai.data.CacheDataset(data=unlabeled_train_files, transform=unlabeled_train_transforms)
                unlabeled_train_loader = monai.data.DataLoader(
                unlabeled_train_ds,
                batch_size=train_bs,
                shuffle=True,
                num_workers=4,
                pin_memory=False, #torch.cuda.is_available(),
                collate_fn=pad_list_data_collate,
            )

            # create a validation data loader for unlabeled data
            print(f"Loading Unlabeled train set - {len(unlabeled_val_files)} patients\n")
            val_unlabeled_transforms = get_kidney_transforms(mode="unlabeled_train")
            val_unlabeled_ds = monai.data.CacheDataset(data=unlabeled_val_files, transform=val_unlabeled_transforms)
            unlabeled_val_loader = monai.data.DataLoader(
                val_unlabeled_ds,
                batch_size=val_bs,  # image-level batch to the sliding window method, not the window-level batch
                num_workers=2,
                collate_fn=pad_list_data_collate,
                pin_memory=False#torch.cuda.is_available(),
            )

            return unlabeled_train_loader, unlabeled_val_loader
        
    elif task_name == "srm":

        keys = ("img", "seg")

        if inference_mode == True:
            
            # Find NRRD files for images and segments
            infer_images, masks  = find_srm_files(infer_data_path)  

            # Print the list of images and segments
            print("Number of inference images:")
            print(len(infer_images))

            infer_images = sorted(infer_images)
            infer_patient_folders = [folder for folder in os.listdir(infer_data_path)
                                            if folder.startswith("Renal-CHUS-") and 
                                            os.path.exists(os.path.join(infer_data_path, folder, "CECT"))] 

            print(f"Inference: total images: ({len(infer_patient_folders)}) from folder: {infer_data_path}")

            # dataloader for test set
            infer_files = [{keys[0]: img, keys[1]: seg} for img, seg in zip(infer_images, masks)]
            


            # create a validation data loader for unlabeled data
            print(f"Loading inference set - {len(infer_files)} patients\n")
            infer_transforms = get_srm_transforms("val", keys=(keys[0], keys[1]))
            infer_ds = monai.data.CacheDataset(data=infer_files, transform=infer_transforms)
            infer_loader = monai.data.DataLoader(
                infer_ds,
                batch_size=1,  # image-level batch to the sliding window method, not the window-level batch
                num_workers=2,
                collate_fn=pad_list_data_collate,
                pin_memory=False#torch.cuda.is_available(),
            )

            return infer_loader
        
        train_frac, val_frac = 0.8, 0.2

            
        # Find NRRD files for images and segments
        labeled_images, labeled_labels = find_srm_files(labeled_data_path)

        # Print the list of images and segments
        print("Number of labeled images:")
        print(len(labeled_images))

        print("\nNumber of labeled segment:")
        print(len(labeled_labels))

        # dataset - labeled data and unlabeled data
        labeled_images = sorted(labeled_images)
        labeled_labels = sorted(labeled_labels)
        labeled_patient_folders = [folder for folder in os.listdir(labeled_data_path)
                                        if folder.startswith("Renal-CHUS-") and 
                                        os.path.exists(os.path.join(labeled_data_path, folder, "CECT"))]
                                        

        print(f"training: total labeled image/label: ({len(labeled_patient_folders)}) from folder: {labeled_data_path}")
        
        # dataloader - for unlabeled cases
        n_train = int(train_frac * len(labeled_images)) + 1
        n_val = min(len(labeled_images) - n_train, int(val_frac * len(labeled_images)))
        print(f"split: labeled train {n_train} val {n_val}, folder: {labeled_data_path}")

        train_files = [{keys[0]: img, keys[1]: seg} for img, seg in zip(labeled_images[:n_train], labeled_labels[:n_train])]
        val_files = [{keys[0]: img, keys[1]: seg} for img, seg in zip(labeled_images[-n_val:], labeled_labels[-n_val:])]
        
        # create a training data loader for labeled data
    
        print(f"Loading Labeled train set - {len(train_files)} patients\n")
        labeled_train_transforms = get_srm_transforms("labeled_train", keys=(keys[0], keys[1]))
        train_ds = monai.data.CacheDataset(data=train_files, transform=labeled_train_transforms)
        labeled_sampler = InfiniteSampler(list_indices=list(range(len(train_files))), batch_size=train_bs)
        labeled_train_loader = monai.data.DataLoader(
            train_ds,
            num_workers=4,
            pin_memory=False,#torch.cuda.is_available(),
            batch_sampler=labeled_sampler
        )

        # create a validation data loader for labeled data

        print(f"Loading Labeled validation set - {len(val_files)} patients\n")
        val_transforms = get_srm_transforms("val", keys=(keys[0], keys[1]))
        val_ds = monai.data.CacheDataset(data=val_files, transform=val_transforms)
        labeled_val_loader = monai.data.DataLoader(
            val_ds,
            batch_size=1,  # image-level batch to the sliding window method, not the window-level batch
            num_workers=2,
            pin_memory=False, #torch.cuda.is_available(),
            collate_fn=pad_list_data_collate,
        )

        return labeled_train_loader, labeled_val_loader


    else:
        assert("Enter a valid task name !!! It should be: kidney or srm.")

    



########## IMAGE-BASED LOADER FOR CLASSIFICATION #############


def load_image_data(task_name, clf_name):

    """
    Data Loader for classification tasks.

    Parameters
    ---
    task_name: there are only 3 tasks: "ccRCC_vs_non_ccRCC", "grade", "subtype".
    clf_name: "cnn" or "xgboost". The augmentation rate differs for each classifier, while cnn requires a high rate, xgboost doesn't.
    """

    batch_size = 32

    if task_name == "ccRCC_vs_non_ccRCC":

        data, target = prepare_classification_data("ccRCC_vs_non_ccRCC")
        if clf_name == "xgb":
            pos_dataset, aug_pos_dataset, neg_dataset, aug_neg_dataset = pos_neg_aug_datasets(data, target, 2, 9)
        elif clf_name == "cnn":
            pos_dataset, aug_pos_dataset, neg_dataset, aug_neg_dataset = pos_neg_aug_datasets(data, target, 4, 14)
        else:
            assert("Enter a valid model name !!!")
        # Create DataLoader for original datasets
        loader_positives = DataLoader(pos_dataset, batch_size=batch_size)
        loader_positives_aug = DataLoader(aug_pos_dataset, batch_size=batch_size)  # additional pos data 
        loader_negatives = DataLoader(neg_dataset, batch_size=batch_size)
        loader_negatives_aug = DataLoader(aug_neg_dataset, batch_size=batch_size)  # additional neg data

        pos_data_dict_ , aug_pos_data_dict_, neg_data_dict_, aug_neg_data_dict_ = dict(), dict(), dict(), dict()

        pos_data_dict_ = get_clf_data_dict(loader_positives)
        aug_pos_data_dict_ = get_clf_data_dict(loader_positives_aug)
        neg_data_dict_ = get_clf_data_dict(loader_negatives)
        aug_neg_data_dict_ = get_clf_data_dict(loader_negatives_aug)

        # Splitting positive data into train and validation/test sets
        X_train_pos, X_val_pos, y_train_pos, y_val_pos, id_train_pos, id_val_pos = train_test_split(
            pos_data_dict_['img'],
            pos_data_dict_['label'],
            pos_data_dict_['Patient_ID'],
            stratify=pos_data_dict_['label'],
            test_size=0.2,
            random_state=42
        )

        # Merging positive training data with augmented and negative data
        X_train_cnn = X_train_pos  + X_val_pos + neg_data_dict_['img']
        y_train_cnn = y_train_pos  + y_val_pos + neg_data_dict_['label']
        id_train_cnn = id_train_pos + id_val_pos + neg_data_dict_['Patient_ID']


        return X_train_cnn, y_train_cnn, id_train_cnn, aug_pos_data_dict_, aug_neg_data_dict_, pos_data_dict_, neg_data_dict_
    
    elif task_name == "grade":

        data, target = prepare_classification_data("grade")
        if clf_name == "xgb":
            pos_dataset, aug_pos_dataset, neg_dataset, aug_neg_dataset = pos_neg_aug_datasets(data, target, 5, 4)
        elif clf_name == "cnn":
            pos_dataset, aug_pos_dataset, neg_dataset, aug_neg_dataset = pos_neg_aug_datasets(data, target, 9, 7)
        else:
            assert("Enter a valid model name !!!")

        # Create DataLoader for original datasets
        loader_positives = DataLoader(pos_dataset, batch_size=batch_size)
        loader_positives_aug = DataLoader(aug_pos_dataset, batch_size=batch_size)  # additional pos data 
        loader_negatives = DataLoader(neg_dataset, batch_size=batch_size)
        loader_negatives_aug = DataLoader(aug_neg_dataset, batch_size=batch_size)  # additional neg data

        pos_data_dict_ , aug_pos_data_dict_, neg_data_dict_, aug_neg_data_dict_ = dict(), dict(), dict(), dict()

        pos_data_dict_ = get_clf_data_dict(loader_positives)
        aug_pos_data_dict_ = get_clf_data_dict(loader_positives_aug)
        neg_data_dict_ = get_clf_data_dict(loader_negatives)
        aug_neg_data_dict_ = get_clf_data_dict(loader_negatives_aug)

        
        # Splitting positive data into train and validation/test sets
        X_train_pos, X_val_pos, y_train_pos, y_val_pos, id_train_pos, id_val_pos = train_test_split(
            pos_data_dict_['img'],
            pos_data_dict_['label'],
            pos_data_dict_['Patient_ID'],
            stratify=pos_data_dict_['label'],
            test_size=0.2,
            random_state=42
        )

        # Merging positive training data with augmented and negative data
        X_train_cnn = X_train_pos  + neg_data_dict_['img']
        y_train_cnn = y_train_pos + neg_data_dict_['label']
        id_train_cnn = id_train_pos + neg_data_dict_['Patient_ID']


        return  X_train_cnn, y_train_cnn, id_train_cnn, aug_pos_data_dict_, aug_neg_data_dict_, pos_data_dict_, neg_data_dict_
    
    elif task_name == "subtype":
        data, target = prepare_classification_data("subtype")
        if clf_name == "xgb":
            pos_dataset, aug_pos_dataset, neg_dataset, aug_neg_dataset = pos_neg_aug_datasets(data, target, 2, 14)
        elif clf_name == "cnn":
            pos_dataset, aug_pos_dataset, neg_dataset, aug_neg_dataset = pos_neg_aug_datasets(data, target, 4, 24)
        else:
            assert("Enter a valid model name !!!")
        # Create DataLoader for original datasets
        loader_positives = DataLoader(pos_dataset, batch_size=batch_size)
        loader_positives_aug = DataLoader(aug_pos_dataset, batch_size=batch_size)  # additional pos data 
        loader_negatives = DataLoader(neg_dataset, batch_size=batch_size)
        loader_negatives_aug = DataLoader(aug_neg_dataset, batch_size=batch_size)  # additional neg data

        pos_data_dict_ , aug_pos_data_dict_, neg_data_dict_, aug_neg_data_dict_ = dict(), dict(), dict(), dict()

        pos_data_dict_ = get_clf_data_dict(loader_positives)
        aug_pos_data_dict_ = get_clf_data_dict(loader_positives_aug)
        neg_data_dict_ = get_clf_data_dict(loader_negatives)
        aug_neg_data_dict_ = get_clf_data_dict(loader_negatives_aug)


        # Splitting positive data into train and validation/test sets
        X_train_pos, X_val_pos, y_train_pos, y_val_pos, id_train_pos, id_val_pos = train_test_split(
            pos_data_dict_['img'],
            pos_data_dict_['label'],
            pos_data_dict_['Patient_ID'],
            stratify=pos_data_dict_['label'],
            test_size=0.2,
            random_state=42
        )


        # Merging positive training data with augmented and negative data
        X_train_cnn = X_train_pos + X_val_pos + neg_data_dict_['img']
        y_train_cnn = y_train_pos + y_val_pos + neg_data_dict_['label']
        id_train_cnn = id_train_pos + id_val_pos + neg_data_dict_['Patient_ID']


        
        return  X_train_cnn, y_train_cnn, id_train_cnn, aug_pos_data_dict_, aug_neg_data_dict_, pos_data_dict_, neg_data_dict_
    
    else:
        assert("Enter a valid task name !!!")


##################### RADIOMICS-BASED LOADER FOR CLASSIFICATION #########################


def load_radiomics_data(aug_pos_data_dict, aug_neg_data_dict, pos_data_dict, neg_data_dict, num_pos, num_neg):
    # Get random subsets
    aug_pos_subset, aug_neg_subset = get_random_subset(aug_pos_data_dict, aug_neg_data_dict, num_pos, num_neg)
    
    pos_features = [pos[:3] for pos in pos_data_dict['radiomics']]
    neg_features = [neg[:3] for neg in neg_data_dict['radiomics']]
    aug_pos_features = [aug_pos[:3] for aug_pos in aug_pos_subset['radiomics']]
    aug_neg_features = [aug_neg[:3] for aug_neg in aug_neg_subset['radiomics']]
        

    X_train_pos, X_val_pos, y_train_pos, y_val_pos, id_train_pos, id_val_pos = train_test_split(
        pos_features, 
        pos_data_dict['label'],
        pos_data_dict['Patient_ID'],
        stratify=pos_data_dict['label'],
        test_size=0.2,
        random_state=42
        )

            
    X_train_xgb = np.concatenate((X_train_pos, X_val_pos), axis=0)
    y_train_xgb = np.concatenate((y_train_pos, y_val_pos), axis=0)
    id_train_xgb = np.concatenate((id_train_pos, id_val_pos), axis=0)

    X_train_xgb = np.concatenate((X_train_xgb, np.array(neg_features)), axis=0)
    y_train_xgb = np.concatenate((y_train_xgb, np.array(neg_data_dict['label'])), axis=0)
    id_train_xgb = np.concatenate((id_train_xgb,  np.array(neg_data_dict['Patient_ID'])), axis=0)

    X_train_xgb_aug = np.concatenate((X_train_xgb, np.array(aug_pos_features)), axis=0)
    y_train_xgb_aug = np.concatenate((y_train_xgb, np.array(aug_pos_subset['label'])), axis=0)
    id_train_xgb_aug = np.concatenate((id_train_xgb,  np.array(aug_pos_subset['Patient_ID'])), axis=0)

    X_train_xgb_aug = np.concatenate((X_train_xgb_aug, np.array(aug_neg_features)), axis=0)
    y_train_xgb_aug = np.concatenate((y_train_xgb_aug, np.array(aug_neg_subset['label'])), axis=0)
    id_train_xgb_aug = np.concatenate((id_train_xgb_aug,  np.array(aug_neg_subset['Patient_ID'])), axis=0)


    # Shuffle the training set
    X_train_xgb, y_train_xgb = shuffle(X_train_xgb, y_train_xgb, random_state=42)

    return X_train_xgb, y_train_xgb, X_train_xgb_aug, y_train_xgb_aug

