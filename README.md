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

- **Features**:
  - `TiffDataset`: Custom dataset class for `.tif` images.
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

## **How to Use**

### **1. Training the Model**
1. Prepare your `.tif` dataset:
   - Place files in a folder (e.g., `./data/`).
   - Use the naming format: `image_<class>.tif` (e.g., `image_1.tif`).
   - Class numbers `1` and `2` are mapped to label `1`, all others to `0`.

2. Set the dataset path in `train.py`:
   ```python
   path = "./path_to_your_dataset"
