import argparse
import os, sys
import xgboost as xgb
import pandas as pd
sys.path.append(os.path.abspath(os.path.join('..', '..')))

# dependencies for kMT and srmUNet Training
from data.dataloaders import load_segmentation_data, load_image_data, load_radiomics_data
from data.dataset import SRMDataset
from models.segmentation.semi_supervised import kidney_segmentor, srm_segmentor
from models.classification.classifier import cnn_classifier, xgboost_classifier
from models.classification.cnn import PatNET
#from utils import save_figures_and_show
from evaluation.classification.evaluate import bootstrap_ci, get_xgb_predictions

# dependencies for bounding box coordinates
from utils import *
from data.roi import load_inference_data, load_labeled_data, load_bbox 
from data.transforms import get_bbox_transforms

def main():
    parser = argparse.ArgumentParser(description="SRM Detection Pipeline CLI")
    subparsers = parser.add_subparsers(dest="command", help="Commands: train, infer, evaluate")

    # kidney MT Training Command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--task", type=str, choices=["segmentation", "classification"], required=True,
                              help="Task type: segmentation or classification.")
    train_parser.add_argument("--seg-task-name", type=str, choices=["kidney", "srm"], 
                              help="Segmentation task type (ignored for classification).")
    train_parser.add_argument("--labeled", type=bool, 
                              help="Specify if Labeled or Unlabeled Dataset (ignored for classification).")
    train_parser.add_argument("--type", type=str, choices=["ccRCC_vs_non_ccRCC", "subtype", "grade"],
                              help="Classification task type (ignored for segmentation).")
    train_parser.add_argument("--data-dir", type=str, required=True, help="Path to the training data.")
    train_parser.add_argument("--output-dir", type=str, required=True, help="Path to save models and results.")
    train_parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    train_parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training.")

    # Inference Command with MT to detect the Region Of Inteest (ROI)
    getroi_parser = subparsers.add_parser("getroi", help="Run inference using pre-trained Mean Teacher to segment the ROI")
    getroi_parser.add_argument("--model-path", type=str, required=True, help="Path to the trained model.")
    getroi_parser.add_argument("--data-dir", type=str, required=True, help="Path to the input data.")
    getroi_parser.add_argument("--bbox-dir", type=str, required=True, help="Path to save box coordinates predictions.")
    getroi_parser.add_argument("--roi-dir", type=str, help="Path to save prediction maps for the ROI.")
    getroi_parser.add_argument("--labeled", type=bool, required=True, help="Does inference set has labels ?")

    
    # Generate bounding boxes for kMT predictions
    bbox_parser = subparsers.add_parser("bbox", help="Generate Bounding Boxes")
    bbox_parser.add_argument("--data-dir", type=str, required=True, help="Path to the input data")
    bbox_parser.add_argument("--bbox-dir", type=str, required=True, help="Path to save the bounding box coordinates")
    bbox_parser.add_argument("--labels-file", type=str, required=True, help="Path to the data")
    bbox_parser.add_argument("--gifs-dir", type=str, help="Path to the save the output as gifs")

    # Classification Inference Command
    classify_parser = subparsers.add_parser("classify", help="Run inference using pre-trained CNN or XGBOOST to classify the tumor")
    classify_parser.add_argument("--task", type=str, choices=["ccRCC_vs_non_ccRCC", "subtype", "grade"],
                               help="Classification task type (ignored for segmentation).")
    classify_parser.add_argument("--model-type", type=str, choices=["cnn", "xgboost", "ConvXgboost"],
                               help="Classification model type (ignored for segmentation).")
    classify_parser.add_argument("--model-path", type=str, required=True, help="Path to the trained model.")
    classify_parser.add_argument("--data-dir", type=str, required=True, help="Path to the input data.")
    classify_parser.add_argument("--output-dir", type=str, required=True, help="Path to save predictions.")

    # SRM Segmentation Inference Command
    segment_parser = subparsers.add_parser("segment", help="Run inference using pre-trained 3D UNet to segment the small renal masses")
    segment_parser.add_argument("--model-path", type=str, required=True, help="Path to the trained model.")
    segment_parser.add_argument("--data-dir", type=str, required=True, help="Path to the input data.")
    segment_parser.add_argument("--output-dir", type=str, required=True, help="Path to save predictions.")

    args = parser.parse_args()


    if args.command == "train":
        if args.task == "segmentation":  #kMT
            
            # Extract labeled and unlabeled data from path
            labeled_train_loader, labeled_val_loader = load_segmentation_data(dataset=args.labeled, task_name=args.seg_task_name, labeled_data_path=args.data_dir)
            unlabeled_train_loader, unlabeled_val_loader = load_segmentation_data(dataset=args.labeled, task_name=args.seg_task_name, unlabeled_data_path=args.data_dir)

            val_interval, epoch_loss_values, metric_values, metric_values_kidney, metric_values_tumor = kidney_segmentor(labeled_train_loader, 
                                                                                                        labeled_val_loader, 
                                                                                                        unlabeled_train_loader)

            #save_figures_and_show(val_interval, epoch_loss_values, metric_values, metric_values_kidney, metric_values_tumor)

        elif args.task == "srm_segmentation": #srmMT
            # Extract labeled and unlabeled data from path
            labeled_train_loader, labeled_val_loader = load_segmentation_data(dataset=args.labeled, task_name=args.seg_task_name, labeled_data_path=args.data_dir)
            unlabeled_train_loader, unlabeled_val_loader = load_segmentation_data(dataset=args.labeled, task_name=args.seg_task_name, unlabeled_data_path=args.data_dir)

            val_interval, epoch_loss_values, metric_values, metric_values_kidney, metric_values_tumor = srm_segmentor(labeled_train_loader, 
                                                                                                        labeled_val_loader, 
                                                                                                        unlabeled_train_loader)

            #save_figures_and_show(val_interval, epoch_loss_values, metric_values, metric_values_kidney, metric_values_tumor)

        else:  
            #train_classification(args.type, args.data_dir, args.output_dir, args.epochs, args.batch_size)
            print("classification")


    elif args.command == "getroi":
         
        infer_loader = load_segmentation_data(task_name="kidney", 
                                                labeled=args.labeled, 
                                                infer_data_path=args.data_dir,  # single patient folder
                                                inference_mode=True)
        
        roi_preds = kidney_segmentor(inference_mode=True,
                                    weights=args.model_path,
                                    inference_loader=infer_loader,
                                    infer_output=args.roi_dir)  # infer_loader has image and ground truth
        # necessary step : collapse prediction
        print(roi_preds)
        # roi_preds should be automatically saved as "mt_prediction.seg.nrrd" in the root folder
        # save_rois(roi_preds, args.output_dir)
        # generate bounding boxes after saving the MT predictions and save them in output_dir
        test_data  = load_inference_data(args.data_dir)
        test_dataset = SRMDataset(data=test_data, data_rcc=pd.read_csv(args.labels_file)['labels'].tolist()) 
        test_loader = DataLoader(test_dataset, batch_size=1)
        # returns dict with img, patient id, label and radiomics
        _, post_pred_transform = get_bbox_transforms()
        test_dataset = load_bbox(test_data, post_pred_transform) 
        save_dict_as_csv(test_dataset, args.bbox_dir)

    elif args.command == "bbox":
        
        test_data  = load_inference_data(args.data_dir)
        test_dataset = SRMDataset(data=test_data, data_rcc=pd.read_csv(args.labels_file)['labels'].tolist()) 
        test_loader = DataLoader(test_dataset, batch_size=1)
        # returns dict with img, patient id, label and radiomics
        post_proc_transform, post_pred_transform = get_bbox_transforms()
        test_dataset = load_bbox(test_data, post_pred_transform) 
        save_dict_as_csv(test_dataset, args.bbox_dir)
        if args.save_gif == True:
            save_gifs(test_data, post_proc_transform, post_pred_transform, output_folder=args.gifs_dir, tag='labeled')


    elif args.command == "train_clf":
        if args.model_type =="cnn": 
            if args.task_name == "ccRCC_vs_non_ccRCC":
                X_train_cnn, y_train_cnn, id_train_cnn, X_test_cnn, y_test_cnn, id_test_cnn, aug_pos_data_dict_, aug_neg_data_dict_, pos_data_dict_, neg_data_dict_ = load_image_data("ccRCC_vs_non_ccRCC", "cnn", inference_mode=True)
                all_train_losses, all_val_losses, all_train_accuracies, all_val_accuracies, best_thresholds, best_roc, fold_indices, best_model = cnn_classifier(task_name="ccRCC_vs_non_ccRCC",
                                                                                                                                                                 model_folder=args.model_path,
                                                                                                                                                                 inference_mode=True,                                                                                                                                                          infer_loader=None,
                                                                                                                                                                 infer_output=args.output_dir)
            elif args.task_name == "grade":
                X_train_cnn, y_train_cnn, id_train_cnn, X_test_cnn, y_test_cnn, id_test_cnn, aug_pos_data_dict_, aug_neg_data_dict_, pos_data_dict_, neg_data_dict_ = load_image_data("grade", "cnn")
                all_train_losses, all_val_losses, all_train_accuracies, all_val_accuracies, best_thresholds, best_roc, fold_indices, best_model = cnn_classifier(inference_mode=True)
            elif args.task_name == "subtype":
                X_train_cnn, y_train_cnn, id_train_cnn, X_test_cnn, y_test_cnn, id_test_cnn, aug_pos_data_dict_, aug_neg_data_dict_, pos_data_dict_, neg_data_dict_ = load_image_data("subtype", "cnn")
                all_train_losses, all_val_losses, all_train_accuracies, all_val_accuracies, best_thresholds, best_roc, fold_indices, best_model = cnn_classifier()
            else:
                print("Invalid task name !!!")
        elif args.model_type == "xgb":
            if args.task_name == "ccRCC_vs_non_ccRCC":
                X_train_xgb, y_train_xgb, X_test_xgb, y_test_xgb, X_train_xgb_aug, y_train_xgb_aug = load_radiomics_data("ccRCC_vs_non_ccRCC")
                model_xgb = xgboost_classifier()
            elif args.task_name == "grade":
                X_train_xgb, y_train_xgb, X_test_xgb, y_test_xgb, X_train_xgb_aug, y_train_xgb_aug = load_radiomics_data("grade")
                model_xgb = xgboost_classifier()
            elif args.task_name == "subtype":
                X_train_xgb, y_train_xgb, X_test_xgb, y_test_xgb, X_train_xgb_aug, y_train_xgb_aug = load_radiomics_data("subtype")
                model_xgb = xgboost_classifier()
            
            else:
                print("Invalid task name !!!")
        elif args.model_type == "conxgb":
            xgboost_classifier()
        else:
            print("Invalid model type !!!")

    
    if args.command == "classify":
                test_data  = load_inference_data(args.data_dir)
                SRMDataset(data=test_data, data_rcc=pd.read_csv(args.labels_file)['label'].tolist()) 
                
                xgb_clf = xgb.XGBClassifier()
                xgb_clf.load_model(args.model_path)
                xgb_probs = get_xgb_predictions(xgb_clf, X_test_xgb)
    
    else:
        print("Wrong command !!!")
if __name__ == "__main__":
    main()
