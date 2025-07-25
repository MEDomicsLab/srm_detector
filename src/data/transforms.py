"""
    @file:              transforms.py
    @Author:            Moustafa Amine Bezzahi, Ihssene Brahimi 

    @Creation Date:     05/2024
    @Last modification: 07/2024

    @Description:       This file is used to define transforms.
"""

import numpy as np
import itertools
import monai.transforms as mt

from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    RandAffined,
    RandAffine,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandFlip,
    RandGaussianNoised,
    ScaleIntensityRanged,
    Spacingd,
    Spacing,
    SpatialPadd,
    ResampleToMatchd,
    LabelToContourd,
    KeepLargestConnectedComponentd,
    FillHolesd,
    RandSpatialCropSamplesd,
    MapLabelValued,
    MapLabelValue
)

from torch.utils.data.sampler import Sampler

######## KIDNEY/TUMOUR SEGMENTATION TASK ##########

# %% transforms: different for train, validation and inference
def get_kidney_transforms(mode="unlabeled_train", keys=("img", "seg", "pred")):  # This is a modified version that contains the transforms for prediction samples
    """Get specefic transforms for labeled and unlabeled datasets"""
    xforms = [
        LoadImaged(keys, allow_missing_keys=False),
        EnsureChannelFirstd(keys),
        Spacingd(keys={keys[0], keys[1]}, pixdim=(2.0, 2.0, 5.0), mode=("bilinear", "nearest")[: len(keys)]), #8 or 5 instead of 10.0
        ScaleIntensityRanged(keys[0], a_min=-500.0, a_max=500.0, b_min=0.0, b_max=1.0, clip=True),
    ]
    if mode == "labeled_train":
        
        xforms.extend(
            [
                SpatialPadd(keys={keys[0], keys[1]}, spatial_size=(128, 128, -1), mode="reflect"),  # ensure at least 192x192
                RandAffined(
                    keys={keys[0], keys[1]},
                    prob=0.15,
                    rotate_range=(0.05, 0.05, None),  # 3 parameters control the transform on 3 dimensions
                    scale_range=(0.1, 0.1, None),
                    mode=("bilinear", "nearest")
                ),
                RandCropByPosNegLabeld(keys={keys[0], keys[1]}, label_key=keys[1], spatial_size=(128, 128, 16), num_samples=3),
                RandGaussianNoised(keys[0], prob=0.15, std=0.01),
                RandFlipd(keys={keys[0], keys[1]}, spatial_axis=0, prob=0.5),
                RandFlipd(keys={keys[0], keys[1]}, spatial_axis=1, prob=0.5),
                RandFlipd(keys={keys[0], keys[1]}, spatial_axis=2, prob=0.5),
            ]
        )
        dtype = (np.float32, np.uint8)
    if mode == "val":
        dtype = (np.float32, np.uint8)
    if mode == "holdout":
        dtype = (np.float32,)
    if mode == "unlabeled_train": 
        xforms.extend(
            [
                ResampleToMatchd(keys={'img', 'seg'}, key_dst ="pred"),
                LabelToContourd(keys={'pred'}),
                KeepLargestConnectedComponentd(keys={'pred'},is_onehot=False, independent=True, connectivity=3, num_components=2),
                FillHolesd(keys={'pred'},applied_labels=[1,2], connectivity=3)
            ]
        )
        dtype = (np.float32,)
    # xforms.extend([CastToTyped(keys, dtype=dtype), EnsureTyped(keys)])
    return Compose(xforms)


def binary_label_transformation(image, segmentation):
    # Convert tumor labels to 1, background remains 0
    transformed_segmentation = np.where(segmentation > 0, 1, 0)
    return image, transformed_segmentation


def get_srm_transforms(mode="unlabeled_train", key="img"):  # This is a modified version that contains the transforms for prediction samples

    if key == "img":
        xforms = []
        
        if mode == "labeled_train":
            
            xforms.extend(
                [
                    RandAffine(
                        prob=0.15,
                        rotate_range=(0.05, 0.05, None),  # 3 parameters control the transform on 3 dimensions
                        scale_range=(0.1, 0.1, None),
                        mode=("bilinear")
                    ),
                    RandFlip(spatial_axis=0, prob=0.5),
                    RandFlip(spatial_axis=1, prob=0.5),
                    RandFlip(spatial_axis=2, prob=0.5),
                ]
            )
            dtype = (np.float32)
        if mode == "val":
            dtype = (np.float32, np.uint8)
        if mode == "holdout":
            dtype = (np.float32,)
        if mode == "unlabeled_train":
            xforms.extend(
                [
                    RandAffine(
                        prob=0.15,
                        rotate_range=(0.05, 0.05, None),  # 3 parameters control the transform on 3 dimensions
                        scale_range=(0.1, 0.1, None),
                        mode=("bilinear")
                    ),
                    RandFlip(spatial_axis=0, prob=0.5),
                    RandFlip(spatial_axis=1, prob=0.5),
                    RandFlip(spatial_axis=2, prob=0.5),
                ]
            )
            dtype = (np.float32, np.uint8)

    else:
        xforms = [

            Spacing(pixdim=(2.0, 2.0, 5.0), mode=("nearest")), #8 or 5 instead of 10.0
            MapLabelValue(
                        orig_labels=np.array([1, 2, 3, 4]),
                        target_labels=np.array([1, 1, 1, 1])),

            ]
    
        if mode == "labeled_train":
            
            xforms.extend(
                [
                    RandAffine(
                        prob=0.15,
                        rotate_range=(0.05, 0.05, None),  # 3 parameters control the transform on 3 dimensions
                        scale_range=(0.1, 0.1, None),
                        mode=("nearest")
                    ),
                    RandFlip(spatial_axis=0, prob=0.5),
                    RandFlip(spatial_axis=1, prob=0.5),
                    RandFlip(spatial_axis=2, prob=0.5),
                ]
            )
            dtype = (np.uint8)
        if mode == "val":
            dtype = (np.uint8)
        if mode == "unlabeled_train":
            xforms.extend(
                [
                    RandAffine(
                        prob=0.15,
                        rotate_range=(0.05, 0.05, None),  # 3 parameters control the transform on 3 dimensions
                        scale_range=(0.1, 0.1, None),
                        mode=("nearest")
                    ),
                    RandFlip(spatial_axis=0, prob=0.5),
                    RandFlip(spatial_axis=1, prob=0.5),
                    RandFlip(spatial_axis=2, prob=0.5),
                ]
            )
            dtype = (np.uint8)
    # xforms.extend([CastToTyped(keys, dtype=dtype), EnsureTyped(keys)])
    return Compose(xforms)

class InfiniteSampler(Sampler):
    def __init__(self, list_indices, batch_size):
        self.indices = list_indices
        self.batch_size = batch_size
    def __iter__(self):
        ss_iter = iterate_eternally(self.indices)
        return (batch for batch in grouper(ss_iter, self.batch_size))
    def __len__(self):
        return len(self.indices)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)


######## BBOX #############
def get_bbox_transforms():

    post_proc = mt.Compose([
        mt.LoadImage(image_only=True, ensure_channel_first=True, simple_keys=True),
        mt.EnsureType(), 
        #mt.Orientation(axcodes='LPS'),  
        mt.Spacing(pixdim=(2.0, 2.0, 5.0), mode=("nearest")),       
        mt.ScaleIntensityRange(a_min=-500.0, a_max=500.0, b_min=0.0, b_max=1.0, clip=True),
        mt.ToTensor()
        ])

    post_pred = mt.Compose([
    mt.LoadImage(image_only=True, ensure_channel_first=True),
    mt.EnsureType(), 
    #mt.Orientation(axcodes='LPS'), 
    mt.Spacing(pixdim=(2.0, 2.0, 5.0), mode=("bilinear")), 
    #mt.ResampleToMatch(),
    mt.LabelToContour(),
    mt.KeepLargestConnectedComponent(is_onehot=False, independent=True, connectivity=3, num_components=2),
    mt.FillHoles(applied_labels=[1,2], connectivity=3),
    mt.ToTensor()
    ])

    return (post_proc, post_pred)
