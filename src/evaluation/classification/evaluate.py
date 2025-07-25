"""
    @file:              evaluate.py
    @Author:            Ihssene Brahimi, Moustafa Amine Bezzahi 

    @Creation Date:     06/2024
    @Last modification: 08/2024

    @Description:       This file is used to define evaluation functions of the classification models.
"""

import numpy as np
import torch
from sklearn.model_selection import train_test_split

import torch
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import resample
import numpy as np

def get_cnn_predictions(model, X, device):
    """
    Get CNN predictions for a dataset without batching.

    Parameters:
    - model: The CNN model.
    - X: A list or array of input data (each element should be a tensor of shape [1, 64, 64, 32]).

    Returns:
    - probs: Predicted probabilities for the positive class.
    """
    model.to(device)
    model.eval()
    all_probs = []

    with torch.no_grad():
        for img in X:
            # Ensure img is a tensor and move it to the appropriate device
            img = img.to(device)
            # Add a batch dimension
            img = img.unsqueeze(0)  # Shape becomes [1, 1, 64, 64, 32]
            outputs, _ = model(img)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            all_probs.append(probs[0])  # probs[0] to get the value from single item

    return all_probs


# Replace with your own trained XGBoost model prediction function
def get_xgb_predictions(model, X):
    return model.predict_proba(X)[:, 1]

def get_cnn_val_data_from_fold_indices(fold_indices, X_train, y_train, id_train, fold):
    """
    Given fold indices, extract validation data for a specific fold.

    Parameters:
    - fold_indices: List of tuples containing training and validation indices for each fold.
    - X_train: The complete training feature dataset.
    - y_train: The complete training label dataset.
    - fold: The index of the fold for which to extract validation data.

    Returns:
    - X_val_cnn: The validation image data for the specified fold.
    - y_val_cnn: The validation label data for the specified fold.
    """
    # Ensure the fold index is valid
    if fold < 0 or fold >= len(fold_indices):
        raise ValueError("Invalid fold index. Must be within the range of available folds.")

    # Get the validation indices for the specified fold
    _, val_index = fold_indices[fold]

    # Extract validation data
    X_val_cnn = [X_train[i] for i in val_index]
    y_val_cnn = [y_train[i] for i in val_index]
    id_val_cnn = [id_train[i] for i in val_index]
    return X_val_cnn, y_val_cnn, id_val_cnn

def get_xgb_val_data_from_fold_indices(fold_indices, X_train, y_train, fold):
    """
    Given fold indices, extract validation data for a specific fold.

    Parameters:
    - fold_indices: List of tuples containing training and validation indices for each fold.
    - X_train: The complete training feature dataset.
    - y_train: The complete training label dataset.
    - fold: The index of the fold for which to extract validation data.

    Returns:
    - X_val_cnn: The validation image data for the specified fold.
    - y_val_cnn: The validation label data for the specified fold.
    """
    # Ensure the fold index is valid
    if fold < 0 or fold >= len(fold_indices):
        raise ValueError("Invalid fold index. Must be within the range of available folds.")

    # Get the validation indices for the specified fold
    _, val_index = fold_indices[fold]

    # Extract validation data
    X_val_xgb = [X_train[i] for i in val_index]
    y_val_xgb = [y_train[i] for i in val_index]

    return X_val_xgb, y_val_xgb

def load_val_xgboost_data(fold, fold_indices, pos_data_dict, neg_data_dict):
    # After CNN training, use the same fold indices for XGBoost training
    pos_features = [pos[:3] for pos in pos_data_dict['radiomics']]
    neg_features = [neg[:3] for neg in neg_data_dict['radiomics']]


    X_train_pos, X_val_test_pos, y_train_pos, y_val_test_pos, id_train_pos, id_val_test_pos = train_test_split(
        pos_features, 
        pos_data_dict['label'],
        pos_data_dict['Patient_ID'],
        stratify=pos_data_dict['label'],
        test_size=0.2,
        random_state=42
    )

    # Splitting the validation/test positive set into validation and test sets
    X_val_pos, X_test_pos, y_val_pos, y_test_pos, id_val_pos, id_test_pos = train_test_split(
        X_val_test_pos,
        y_val_test_pos,
        id_val_test_pos,
        stratify=y_val_test_pos,
        test_size=0.50,
        random_state=42
    )
        
    X_train_xgb = np.concatenate((X_train_pos, np.array(X_val_pos)), axis=0)
    y_train_xgb = np.concatenate((y_train_pos, np.array(y_val_pos)), axis=0)
    id_train_xgb = np.concatenate((id_train_pos, np.array(id_val_pos)), axis=0) 

    X_train_xgb = np.concatenate((X_train_xgb, np.array(neg_features[22:])), axis=0)
    y_train_xgb = np.concatenate((y_train_xgb, np.array(neg_data_dict['label'][22:])), axis=0)
    id_train_xgb = np.concatenate((id_train_xgb, np.array(neg_data_dict['Patient_ID'][22:])), axis=0) 

    X_test_xgb = np.concatenate((X_test_pos, np.array(neg_features[:22])), axis=0)
    y_test_xgb = np.concatenate((y_test_pos, np.array(neg_data_dict['label'][:22])), axis=0)
    id_test_xgb = np.concatenate((id_test_pos, np.array(neg_data_dict['Patient_ID'][:22])), axis=0) 

    _, val_index = fold_indices[fold]

    print(len(val_index))
    print(len(X_train_xgb))
    # Extract validation data
    X_val_xgb = [X_train_xgb[i] for i in val_index]
    y_val_xgb = [y_train_xgb[i] for i in val_index]
    id_val_xgb = [id_train_xgb[i] for i in val_index]
    
    return X_val_xgb, y_val_xgb, id_val_xgb, X_test_xgb, y_test_xgb, id_test_xgb


# Function to evaluate model and generate metrics
def evaluate_model(y_true, y_preds, y_proba):
    cm = confusion_matrix(y_true, y_preds)
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = accuracy_score(y_true, y_preds)
    precision = precision_score(y_true, y_preds)
    recall = recall_score(y_true, y_preds)
    f1 = f1_score(y_true, y_preds)
   # ppv = precision  # Positive Predictive Value is precision
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    specificity = tn / (tn + fp)
    sensitivity = recall
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    return {

        'Confusion Matrix': cm,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Specificity': specificity,
        'Sensitivity': sensitivity,
        'ROC AUC': roc_auc,
        'FPR': fpr,
        'TPR': tpr,
        'PPV': precision,
        'NPV': npv
    }

def bootstrap_auc_confidence_interval(y_true, y_proba, n_bootstraps=1000, alpha=0.95):
    """Calculate the bootstrap AUC confidence interval."""
    bootstrapped_scores = []
    y_true = np.array(y_true)
    y_proba = np.array(y_proba)
    for _ in range(n_bootstraps):
        indices = np.array(resample(range(len(y_true)), replace=True))
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = roc_auc_score(y_true[indices], y_proba[indices])
        bootstrapped_scores.append(score)
    sorted_scores = np.sort(bootstrapped_scores)
    lower_bound = np.percentile(sorted_scores, (1 - alpha) / 2 * 100)
    upper_bound = np.percentile(sorted_scores, (1 + alpha) / 2 * 100)
    return lower_bound, upper_bound

#bootstrapping results 
def bootstrap_ci(values, n_bootstrap=1000, confidence_level=0.95):
    """Calculate the bootstrap confidence interval."""
    bootstrap_samples = []
    for _ in range(n_bootstrap):
        sample = resample(values, replace=True, n_samples=len(values))
        bootstrap_samples.append(sample)

    bootstrap_samples.sort()
    lower = np.percentile(bootstrap_samples, (1 - confidence_level) / 2 * 100)
    upper = np.percentile(bootstrap_samples, (1 + confidence_level) / 2 * 100)
    return lower, upper
