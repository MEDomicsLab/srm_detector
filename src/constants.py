"""
    @file:              constants.py
    @Author:            Ihssene Brahimi, Moustafa Amine Bezzahi 

    @Creation Date:     07/2024
    @Last modification: 07/2024

    @Description:       This file is used to define the Constants paths used in the pipeline.
"""
import pandas as pd
import os


CSV_DIR = "srm_detection_pipeline/output/csv"

UNLABELED_TEST_DATA = pd.read_csv("../output/csv/bbox_unlabeled_test_set.csv")
UNLABELED_TRAIN_DATA = pd.read_csv("../output/csv/bbox_unlabeled_train_set.csv")

LABELED_TEST_DATA = pd.read_csv("../output/csv/bbox_labeled_test_set.csv")
LABELED_TRAIN_DATA = pd.read_csv("../output/csv/bbox_labeled_train_set.csv") 


LABELED_DATA_PATH = 'srm_detection_pipeline/application/dataset/labeled'
UNLABELED_DATA_PATH = 'srm_detection_pipeline/application/dataset/unlabeled'

KIDNEY_SEGMENTATION_MODEL = "srm_detection/models/weights/segmentation/kidney/best_metric_model_full_256.pth"
SRM_SEGMENTATION_MODEL = "srm_detection/models/weights/segmentation/srm/best_fully_supervised_srm_model_128.pth"

XGBOOST_CCRCC_VS_NON_CCRCC_MODEL = "srm_detection_pipeline/srm_detection/models/weights/classification/ccRCC_vs_non_ccRCC/xgboost/xgboost_model_attempt_9_more_aug_600.bin"   
XGBOOST_SUBTYPE_MODEL = "srm_detection_pipeline/srm_detection/models/weights/classification//subtype/xgboost/xgboost_model_attempt_8.bin"
XGBOOST_GRADE_MODEL = "srm_detection_pipeline/srm_detection/models/weights/classification/grade/xgboost/xgboost_model_attempt_5.bin"


CNN_CCRCC_VS_NON_CCRCC_MODEL = "srm_detection_pipeline/srm_detection/models/weights/classification/ccRCC_vs_non_ccRCC/patnet/best_ccRCC_vs_non_ccRCC_clf_fold_5.pth"
CNN_SUBTYPE_MODEL = "srm_detection_pipeline/srm_detection/models/weights/classification//subtype/patnet/best_subtype_clf_fold_5.pth"
CNN_GRADE_MODEL = "srm_detection_pipeline/srm_detection/models/weights/classification/grade/patnet/best_grade_clf_fold_5.pth"



SEGMENTATION_OUTPUT_FILES = "srm_detection_pipeline/output/maps"
GIFS_FOLDER = "srm_detection_pipeline/output/gifs"
