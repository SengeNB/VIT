import os
import shutil
import random
from pathlib import Path

def split_images(source_folder, output_folder, train_ratio=0.8, seed=42):
    random.seed(seed)
    source_folder = Path(source_folder)
    train_folder = Path(output_folder) / 'train'
    valid_folder = Path(output_folder) / 'valid'
    train_folder.mkdir(parents=True, exist_ok=True)
    valid_folder.mkdir(parents=True, exist_ok=True)
    
    image_files = sorted(source_folder.glob("*"))  
    random.shuffle(image_files)  
    
    split_idx = int(len(image_files) * train_ratio)
    train_files = image_files[:split_idx]
    valid_files = image_files[split_idx:]
    

    for file in train_files:
        shutil.copy(file, train_folder / file.name)
    
    for file in valid_files:
        shutil.copy(file, valid_folder / file.name)
    
    print(f"Total images: {len(image_files)}")
    print(f"Train images: {len(train_files)}")
    print(f"Valid images: {len(valid_files)}")


source_folder = r""  # replace your image folder address
output_folder = r""  # replace your train and valid folder address
split_images(source_folder, output_folder)
