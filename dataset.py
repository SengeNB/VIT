import os
import tifffile as tiff
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
import numpy as np
import torchvision.transforms as transforms
import cv2

class TiffDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_paths = []
        self.labels = []
        
        for file_name in os.listdir(root_dir):
            if file_name.endswith('.tif'):
                file_path = os.path.join(root_dir, file_name)
                self.file_paths.append(file_path)
                
                try:
                    class_num = int(file_name.split('_')[-1].split('.')[0])
                    label = 1 if class_num in [1, 2] else 0
                except ValueError:
                    continue
                
                self.labels.append(label)
                
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        image = tiff.imread(file_path)
        label = self.labels[idx]
        
        # 提取 target 信息
        target = int(file_path.split('_')[-1].split('.')[0])
        
        if self.transform:
            image = self.transform(image)
        if image.shape != (16, 80, 80):
            image = np.transpose(image, (1, 0, 2))
        if image.shape != (16, 80, 80):
            raise ValueError(f"Unexpected image shape: {image.shape}, expected (16, 80, 80)")
        return image, label, target

    def check_image_names(self):
        for file_path, label in zip(self.file_paths, self.labels):
            file_name = os.path.basename(file_path)
            try:
                class_num = int(file_name.split('_')[-1].split('.')[0])
                if (label == 0 and class_num not in [1, 2, 3]) or (label == 1 and class_num not in [0, 4, 5, 6]):
                    return False
            except ValueError:
                return False
        return True

def random_rotation(image, angle_range=(-30, 30)):
    angle = np.random.uniform(*angle_range)
    return np.array([cv2.rotate(slice, cv2.ROTATE_90_CLOCKWISE) for slice in image])

def random_blur(image, ksize=3):
    return np.array([cv2.GaussianBlur(slice, (ksize, ksize), 0) for slice in image])

def custom_transform(image):
    image = random_rotation(image)
    image = random_blur(image)
    return image

def load_dataset(train_path, valid_path, num_classes=2):
    train_transform = custom_transform
    val_transform = None
    
    train_dataset = TiffDataset(train_path, transform=train_transform)
    val_dataset = TiffDataset(valid_path, transform=val_transform)

    total_train_files = len(train_dataset)
    total_valid_files = len(val_dataset)
    class0_count_train = sum(1 for label in train_dataset.labels if label == 0)
    class1_count_train = sum(1 for label in train_dataset.labels if label == 1)
    class0_count_valid = sum(1 for label in val_dataset.labels if label == 0)
    class1_count_valid = sum(1 for label in val_dataset.labels if label == 1)
    
    print(f"Train files number: {total_train_files}")
    print(f"Valid files number: {total_valid_files}")
    print(f"Train Class 0 files number: {class0_count_train}")
    print(f"Train Class 1 files number: {class1_count_train}")
    print(f"Valid Class 0 files number: {class0_count_valid}")
    print(f"Valid Class 1 files number: {class1_count_valid}")
    
    return train_dataset, val_dataset

def check_image_sizes(dataset, expected_shape=(16, 80, 80)):
    for idx in range(len(dataset)):
        image, label = dataset[idx]
        if image.shape != expected_shape:
            return False
    return True

def check_dataloader_account(loader):
    train_class_counts = count_samples_per_class(loader)
    print(f"训练集每个类别的样本数量: {train_class_counts}")


def count_samples_per_class(data_loader):
    class_counts = {}
    for _, labels, targets in data_loader:
        for label, target in zip(labels, targets):
            label = label.item()
            target = int(target)  
            if label not in class_counts:
                class_counts[label] = {'total': 0, 'targets': {}}
            class_counts[label]['total'] += 1
            if target in class_counts[label]['targets']:
                class_counts[label]['targets'][target] += 1
            else:
                class_counts[label]['targets'][target] = 1
    return class_counts


def collate_fn(batch):
    images, labels, targets = zip(*batch)
    # Convert numpy arrays to tensors and ensure they have the correct shape
    images = [torch.tensor(image).float() / 255.0 for image in images]  
    # Stack images to form a batch
    images = torch.stack(images, dim=0)
    labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)  
    targets = torch.tensor(targets, dtype=torch.float32)  
    return images, labels, targets
