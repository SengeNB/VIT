from dataset import load_dataset
from dataset import check_image_sizes
from model import CustomViT
from model import train_model 
from model import validate_model
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import tifffile as tiff
from dataset import TiffDataset
from dataset import check_dataloader_account
from imagesave import create_run_folder
from imagesave import save_plots
from model import clear_gpu_memory
from dataset import collate_fn
def main():
    
    #path = "./dataset_without3"
    #path = "./dataset_full"
    train_path = r"C:\Users\16074\Desktop\tiger\ALL_DATA\train"  # replace your train file 
    valid_path = r"C:\Users\16074\Desktop\tiger\ALL_DATA\valid"  # replace your valid file 
    train_dataset, val_dataset = load_dataset(train_path, valid_path)
    # dataset = TiffDataset(path)

    # if dataset.check_image_names():
    #     print("所有图像名称均符合要求")
    # else:
    #     print("一些图像名称不符合要求")
    # if check_image_sizes(train_dataset) and check_image_sizes(val_dataset):
    #     print("所有图像尺寸均符合要求")
    # else:
    #     print("一些图像尺寸不符合要求")

    print("------------------------------")
    print("checking cuda")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device is", device)
    print("------------------------------")
    print("Create data loader")
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True,collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False,collate_fn=collate_fn)
    print("Check number loader")
    check_dataloader_account(train_loader)
    check_dataloader_account(val_loader)
    print("Clean GPU")
    clear_gpu_memory()
    print("Create model")
    #pretrainPth = r"C:\Users\16074\Desktop\tiger\pretrained_weights"
    model = CustomViT()
    model = model.to(device)

    #criterion = nn.CrossEntropyLoss()
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=30, verbose=True)
    print("start training:")
    run_folder = create_run_folder()
    num_epochs = 100

    metrics = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs,run_folder=run_folder)
        
    save_plots(run_folder, metrics)


if __name__ == '__main__':
    main()
