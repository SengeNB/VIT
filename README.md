# **ViT-Based Binary Classification for TIF Images**

This repository implements a Vision Transformer (ViT) model for binary classification of multi-channel `.tif` images. It includes functionalities for training, validation, prediction, and dataset preprocessing.

---

## **Features**
1. Train a custom ViT model using `.tif` images.
2. Validate and evaluate model performance on a validation dataset.
3. Predict labels for individual images or entire datasets.
4. Dataset preprocessing with custom augmentations.
5. Automatic logging of metrics and visualizations.

---

## **File Structure**
### 1. `train.py`
Main script for training and validating the model.

- **Key Steps**:
  - Load and preprocess the dataset.
  - Define the training configuration (model, optimizer, loss function, scheduler).
  - Train the model and log metrics.
  - Save the best-performing models and visualizations.

### 2. `model.py`
Defines the Vision Transformer model and training/validation functions.

- **Components**:
  - `CustomViT`: The custom ViT architecture for binary classification.
  - `train_model`: Function for training the model and saving metrics.
  - `validate_model`: Function for evaluating model performance on validation data.
  - `clear_gpu_memory`: Utility for GPU memory management.

### 3. `datasets.py`
Handles dataset loading, preprocessing, and augmentation.

- **Key Features**:
  - **Label Mapping**:
    - In the current implementation, class numbers `1` and `2` are mapped to label `1`, while all other class numbers are mapped to label `0`. This is defined in the `TiffDataset` class.
    - If you need to change this mapping, modify the `TiffDataset` class in `datasets.py`:
      ```python
      label = 1 if class_num in [1, 2] else 0
      ```
  - `custom_transform`: Augmentation logic for data augmentation.
  - `load_dataset`: Splits dataset into training and validation sets.
  - Utilities for checking dataset integrity.


### 4. `predict_all.py`
Predicts labels for an entire dataset and computes evaluation metrics.

- **Metrics**:
  - Accuracy
  - Precision
  - Recall
  - Confidence scores
  - Per-class statistics

### 5. `predict.py`
Predicts the label and confidence score for a single image.

---
### 6. `split_train_valid.py`
Sort all the images in a folder by a fixed seed to ensure reproducible results.
Copying files, rather than moving them, guarantees that the original data set will not be modified. Create the train and valid destination folders and copy the corresponding images into them.

## **How to Use**

### **1. Training the Model**

1. Set the dataset path in `train.py`:
   ```python
   path = "./path_to_your_dataset"
2. Run the training script:
   python train.py
3. Training outputs:
   - Best models: best_val_loss_model.pth and best_val_acc_model.pth.
   - Metrics: training_metrics.csv.
   - Plots: Saved in runs/runX/.

### **2. Making Predictions**
  For an Entire Dataset:
  1. Set the dataset path and model path in predict_all.py:
     model_path = "./path_to_trained_model.pth"
     image_folder = "./path_to_dataset/"
  2. Run the script:
    python predict_all.py
  3. Outputs:
     Accuracy, precision, recall, and average confidence scores.
     
  For a Single Image:
  1. Set the model path in predict.py
     model_path = "./path_to_trained_model.pth"
  2. Run the script:
     python predict.py
  3. Outputs:
     Predicted label and confidence score.
     
