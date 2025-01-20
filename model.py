import torch
from torch import nn
from transformers import ViTModel, ViTConfig
import gc
from sklearn.metrics import precision_score, recall_score
import numpy as np
import os
import torch.optim as optim
from earlystop import EarlyStopping
import pandas as pd

class CustomViT(nn.Module):
    def __init__(self, image_size=(16, 80, 80)):
        super(CustomViT, self).__init__()
        
        self.config = ViTConfig(
            num_channels=image_size[0],  
            image_size=image_size[1],    
            patch_size=16,
            num_labels=1,  
            hidden_size=512,
            num_hidden_layers=16,
            num_attention_heads=16,
            intermediate_size=3072,
            hidden_act="gelu",
            layer_norm_eps=1e-12,
            dropout_rate=0.1,
        )
        
        self.vit = ViTModel(self.config)
        
        self.classifier = nn.Linear(self.config.hidden_size, 1)  

    def forward(self, x):
        assert len(x.shape) == 4, f"Expected input shape (batch_size, num_channels, height, width), got {x.shape}"
        
        outputs = self.vit(pixel_values=x)

        cls_output = outputs.last_hidden_state[:, 0, :]
    
        logits = self.classifier(cls_output)
        probabilities = torch.sigmoid(logits)  # Apply sigmoid to get probabilities
        return probabilities


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=200, run_folder=None):
    metrics = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'train_precision': [],
        'val_precision': [],
        'train_recall': [],
        'val_recall': []
    }
    
    best_val_loss = float('inf')
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_preds = []
        
        for inputs, labels, _ in train_loader: 
            inputs, labels = inputs.to(device), labels.to(device).float().view(-1, 1)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            predicted = outputs.round()  # Outputs are already probabilities
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.detach().cpu().numpy())
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total

        pred_class_distribution = dict(zip(*np.unique(all_preds, return_counts=True)))
        print(f"Epoch {epoch} - Predicted class distribution: {pred_class_distribution}")
        
        label_class_distribution = dict(zip(*np.unique(all_labels, return_counts=True)))
        print(f"Epoch {epoch} - Actual class distribution: {label_class_distribution}")

        epoch_precision = precision_score(all_labels, all_preds, average='weighted', zero_division=1)
        epoch_recall = recall_score(all_labels, all_preds, average='weighted', zero_division=1)
        
        metrics['train_loss'].append(epoch_loss)
        metrics['train_acc'].append(epoch_acc)
        metrics['train_precision'].append(epoch_precision)
        metrics['train_recall'].append(epoch_recall)
        
        print(f'Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}, Precision: {epoch_precision:.4f}, Recall: {epoch_recall:.4f}')
        
        val_loss, val_acc, val_precision, val_recall = validate_model(model, val_loader, criterion, device)
        metrics['val_loss'].append(val_loss)
        metrics['val_acc'].append(val_acc)
        metrics['val_precision'].append(val_precision)
        metrics['val_recall'].append(val_recall)
        
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}, Validation Precision: {val_precision:.4f}, Validation Recall: {val_recall:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(run_folder, 'best_val_loss_model.pth'))
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(run_folder, 'best_val_acc_model.pth'))
        
        scheduler.step(val_loss)

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(os.path.join(run_folder, 'training_metrics.csv'), index=False)
    
    print('Training complete')
    return metrics

def validate_model(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for inputs, labels, _ in val_loader:  # 忽略额外的返回值
            inputs, labels = inputs.to(device), labels.to(device).float().view(-1, 1)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            predicted = outputs.round()  # Outputs are already probabilities
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.detach().cpu().numpy())
    
    val_loss = running_loss / len(val_loader.dataset)
    val_acc = correct / total  # 计算准确率时除以总样本数

    # 打印预测的类别分布
    pred_class_distribution = dict(zip(*np.unique(all_preds, return_counts=True)))
    print(f"Validation - Predicted class distribution: {pred_class_distribution}")
    
    label_class_distribution = dict(zip(*np.unique(all_labels, return_counts=True)))
    print(f"Validation - Actual class distribution: {label_class_distribution}")

    val_precision = precision_score(all_labels, all_preds, average='weighted', zero_division=1)
    val_recall = recall_score(all_labels, all_preds, average='weighted', zero_division=1)
    
    return val_loss, val_acc, val_precision, val_recall

def clear_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()

def create_run_folder(base_folder="runs"):
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    
    run_folders = [f for f in os.listdir(base_folder) if f.startswith("run")]
    run_number = len(run_folders) + 1
    run_folder = os.path.join(base_folder, f"run{run_number}")
    os.makedirs(run_folder)
    
    return run_folder
