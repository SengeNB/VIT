from torch.utils.data import DataLoader
import tifffile as tiff
from dataset import load_dataset
from dataset import TiffDataset
import os
import shutil

# 定义一个函数来检查数据集中的图像形状
def check_image_shape(dataset):
    for image, label in dataset:
        print("Image shape:", image.shape)
        break  # 检查一个图像就可以了
def main():

    root_dir = "./dataset_full"  # 替换为你的文件路径
    suffix_counts = count_file_suffixes(root_dir)

    # 打印统计结果
    print(f"尾数为0的文件数量: {suffix_counts[0]}")
    print(f"尾数为1的文件数量: {suffix_counts[1]}")
    print(f"尾数为2的文件数量: {suffix_counts[2]}")
    print(f"尾数为3的文件数量: {suffix_counts[3]}")
    print(f"尾数为4的文件数量: {suffix_counts[4]}")

def count_file_suffixes(root_dir):
    suffix_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

    # 遍历目录，读取文件名并统计尾数
    for file_name in os.listdir(root_dir):
        if file_name.endswith('.tif'):
            try:
                # 提取文件名中的尾数
                class_num = int(file_name.split('_')[-1].split('.')[0])
                if class_num in suffix_counts:
                    suffix_counts[class_num] += 1
            except ValueError:
                continue
    
    return suffix_counts

def copy_files_exclude_suffix_3(src_dir, dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    for file_name in os.listdir(src_dir):
        if file_name.endswith('.tif'):
            try:
                class_num = int(file_name.split('_')[-1].split('.')[0])
                if class_num == 3:
                    continue  # 跳过尾数为3的文件
            except ValueError:
                continue

            src_file_path = os.path.join(src_dir, file_name)
            dst_file_path = os.path.join(dst_dir, file_name)
            shutil.copy(src_file_path, dst_file_path)
            print(f"Copied {file_name} to {dst_dir}")

if __name__ == '__main__':
    src_dir = "./dataset_full"  # 替换为你的源文件夹路径
    dst_dir = "./dataset_without3"  # 替换为你的目标文件夹路径
    copy_files_exclude_suffix_3(src_dir, dst_dir)