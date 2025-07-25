"""
    @file:              classifier.py
    @Author:            Moustafa Amine Bezzahi, Ihssene Brahimi

    @Creation Date:     06/2024
    @Last modification: 08/2024

    @Description:       This file is used to define the classification models.
"""

from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
import numpy as np
import os
from data.dataset import CNNDataset
from utils import FilteredDataLoader
from xgboost import XGBClassifier
from sklearn.calibration import IsotonicRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from data.dataloaders import load_radiomics_data


def _train(model, train_loader, optimizer, criterion, device, intermediate_output=True):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_labels = []
    train_preds = []

    for train_data in train_loader:
        inputs, labels = train_data['img'].to(device), train_data['label'].to(device)
        optimizer.zero_grad()
        
        if intermediate_output:
            outputs, _ = model(inputs)
        else:
            outputs = model(inputs)
            
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        train_correct += torch.sum(preds == labels.data)
        train_labels.extend(labels.cpu().numpy())
        train_preds.extend(preds.cpu().numpy())

    train_loss = train_loss / len(train_loader.dataset)
    train_accuracy = accuracy_score(train_labels, train_preds)

    return train_loss, train_accuracy

def _evaluate(model, val_loader, device, y_val_fold, intermediate_output=True):
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_labels = []
    val_preds = []
    val_probs = []

    class_counts = torch.bincount(torch.tensor(y_val_fold))
    class_weights = 1. / class_counts.float()
    weights = torch.tensor([class_weights[0], class_weights[1]], dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)

    with torch.no_grad():
        for val_data in val_loader:
            inputs, labels = val_data['img'].to(device), val_data['label'].to(device)
            
            if intermediate_output:
                outputs, _ = model(inputs)
            else:
                outputs = model(inputs)
                
            loss = criterion(outputs, labels)
            probs = nn.Softmax(dim=1)(outputs)
            val_probs.extend(probs[:, 1].cpu().numpy())

            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += torch.sum(preds == labels.data)
            val_labels.extend(labels.cpu().numpy())
            val_preds.extend(preds.cpu().numpy())

    val_loss = val_loss / len(val_loader.dataset)
    val_accuracy = accuracy_score(val_labels, val_preds)

    fpr, tpr, thresholds = roc_curve(val_labels, val_probs)
    roc_auc = auc(fpr, tpr)

    j_scores = tpr - fpr
    best_threshold = thresholds[np.argmax(j_scores)]

    return val_loss, val_accuracy, roc_auc, best_threshold


def cnn_classifier(model, 
                    X_train, 
                    y_train, 
                    id_train, 
                    aug_pos_data_dict, 
                    aug_neg_data_dict, 
                    task_name, 
                    model_folder,
                    max_epochs=10, 
                    patience=3, 
                    n_splits=5, 
                    lr=0.001,
                    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    
   

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    all_train_losses = []
    all_val_losses = []
    all_train_accuracies = []
    all_val_accuracies = []
    best_thresholds = []
    fold_indices = []

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    if task_name in ["ccRCC_vs_non_ccRCC", "grade", "subtype"]:
        aug_pos_imgs = aug_pos_data_dict['img']
        aug_pos_labels = aug_pos_data_dict['label']
        aug_pos_ids = aug_pos_data_dict['Patient_ID']

        aug_neg_imgs = aug_neg_data_dict['img']
        aug_neg_labels = aug_neg_data_dict['label']
        aug_neg_ids = aug_neg_data_dict['Patient_ID']
    else:
        raise ValueError("Enter a valid task name !!!")

    for fold, (train_index, val_index) in enumerate(kf.split(X_train, y_train)):
        print(f"Fold {fold + 1}/{n_splits}")

        fold_indices.append((train_index, val_index))  # save fold indices for ensemble training with xgboost

        best_loss = float('inf')
        patience_counter = 0

        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []

        X_train_fold = [X_train[i] for i in train_index]
        y_train_fold = [y_train[i] for i in train_index]
        id_train_fold = [id_train[i] for i in train_index]

        X_val_fold = [X_train[i] for i in val_index]
        y_val_fold = [y_train[i] for i in val_index]
        id_val_fold = [id_train[i] for i in val_index]

        X_train_fold.extend(aug_pos_imgs)
        y_train_fold.extend(aug_pos_labels)
        id_train_fold.extend(aug_pos_ids)

        X_train_fold.extend(aug_neg_imgs)
        y_train_fold.extend(aug_neg_labels)
        id_train_fold.extend(aug_neg_ids)

        train_fold_dataset = CNNDataset(data=X_train_fold, data_rcc=y_train_fold, patient_ids=id_train_fold)
        val_fold_dataset = CNNDataset(data=X_val_fold, data_rcc=y_val_fold, patient_ids=id_val_fold)

        train_fold_loader = FilteredDataLoader(train_fold_dataset, batch_size=32, shuffle=True)
        val_fold_loader = FilteredDataLoader(val_fold_dataset, batch_size=8, shuffle=True)


        for epoch in range(max_epochs):
            train_loss, train_accuracy = _train(model, train_fold_loader, optimizer, criterion, device)
            val_loss, val_accuracy, roc_auc, best_threshold = _evaluate(model, val_fold_loader, device, y_val_fold)

            print(f'Epoch: {epoch+1} \t Training Loss: {train_loss:.5f} \t Training Accuracy: {train_accuracy:.5f} \t Validation Loss: {val_loss:.5f} \t Validation Accuracy: {val_accuracy:.5f}')
            print(f'Best roc/auc threshold for fold {fold + 1}: {best_threshold}')

            best_thresholds.append(best_threshold)

            scheduler.step(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                best_model = model
                best_roc = roc_auc
                torch.save(model.state_dict(), os.path.join(model_folder, f'{task_name}/best_{task_name}_clf_fold_{fold + 1}.pth'))
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print("Early stopping triggered")
                break

            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)
        all_train_accuracies.append(train_accuracies)
        all_val_accuracies.append(val_accuracies)

    return all_train_losses, all_val_losses, all_train_accuracies, all_val_accuracies, best_thresholds, best_roc, fold_indices, best_model




def _train_xgb(X_train, 
            X_test, 
            y_train, 
            y_test,
            var_importance_threshold: float = 0.01, 
            cv=5, 
            calibrate=True):
    
    # Train XGBoost model
    classifier = XGBClassifier()
    classifier.fit(X_train, y_train)

    # Calculate feature importance
    var_importance = classifier.feature_importances_
    
    # Normalize var_importance if necessary
    if np.sum(var_importance) != 1:
        var_importance = var_importance / np.sum(var_importance)
    
    # Filter variables based on importance threshold
    selected_features = var_importance >= var_importance_threshold
    
    # Check if any features are selected
    if np.sum(selected_features) == 0:
        raise ValueError('No features selected. Please use a smaller threshold.')

    # Select columns based on selected_features
    X_train_filtered = X_train[:, selected_features]
    X_test_filtered = X_test[:, selected_features]
    
    
    if calibrate:
        # Predictions on test set
        x_prob = classifier.predict_proba(X_train_filtered)[:, 1]
        claibrater = IsotonicRegression(out_of_bounds='clip')
        claibrater.fit(x_prob, y_train)

    # XGB Classifier with suggested scale_pos_weight
    scale_pos_weight = (len(y_train) - np.sum(y_train)) / np.sum(y_train)
    classifier = XGBClassifier(scale_pos_weight=scale_pos_weight)
    
    # Tune XGBoost parameters
    params = {
    "max_depth" : [7, 9,11],
    "learning_rate" : [0.25,0.3],
    "colsample_bytree" : np.arange(0.7,1,0.1),
    "subsample": np.arange(0.7,1,0.1),
    "gamma" : [0.1,0.2],
    "n_estimators": [100, 200, 300, 500]
    }  
    
    # Grid search
    grid_search = GridSearchCV(estimator=classifier, param_grid=params, scoring='accuracy', cv=cv)
    grid_search.fit(X_train_filtered, y_train)
    
    # Get the best classifier from grid search
    best_classifier = grid_search.best_estimator_
    
    # Predictions on test set
    y_prob = best_classifier.predict_proba(X_test_filtered)[:, 1]
    y_prob_calib = claibrater.transform(y_prob)
    fpr, tpr, _ = roc_curve(y_test, y_prob_calib)
    roc_auc = auc(fpr, tpr)

    
    # Confusion Matrix
    y_pred = best_classifier.predict(X_test_filtered)
    cm = confusion_matrix(y_test, y_pred)

    
    # Return model information
    model_xgb = {
        'model': best_classifier,
        'selected_features': selected_features,
        'best_params': grid_search.best_params_,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc,
        'cm': cm, 
        'fpr': fpr,
        'tpr': tpr
    }
    
    return model_xgb




def xgboost_classifier(task_name, 
                    model_folder, 
                    aug_pos_data_dict, 
                    aug_neg_data_dict, 
                    pos_data_dict, 
                    neg_data_dict, 
                    num_pos, 
                    num_neg):
    
        #for attempt in range(num_attempts):
        
        X_train_xgb, y_train_xgb, X_test_xgb, y_test_xgb, X_train_xgb_aug, y_train_xgb_aug = load_radiomics_data(aug_pos_data_dict, 
                                                                                                               aug_neg_data_dict, 
                                                                                                               pos_data_dict, 
                                                                                                               neg_data_dict, 
                                                                                                               num_pos,                                                                                                    
                                                                                                               num_neg)        
        # Train XGBoost model
        model_xgb = _train_xgb(X_train_xgb_aug, X_test_xgb, y_train_xgb_aug, y_test_xgb) 
        print(model_xgb)

        # Save the model
        model_xgb['model'].save_model(os.path.join(model_folder, f"xgboost_model_{task_name}.bin"))
        
        print(f"Model saved to {model_folder}")

        return model_xgb


