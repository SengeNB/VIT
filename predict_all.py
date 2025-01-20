import torch
import os
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.metrics import accuracy_score, recall_score, precision_score
from model import CustomViT
from dataset import TiffDataset
from collections import defaultdict

# 加载模型
def load_model(model_path, device):
    model = CustomViT().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# 预测函数
def predict_images(model, image_folder, device):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = TiffDataset(image_folder, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    all_confidences = []
    
    stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'incorrect': 0})

    for images, labels, targets in dataloader:
        images = images.to(device).float()
        labels = labels.to(device).float().view(-1, 1)

        with torch.no_grad():
            outputs = model(images)
            probabilities = outputs.cpu().numpy()  # Model output is already probabilities
            predicted = (probabilities > 0.5).astype(int)  # Convert probabilities to binary predictions
        
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted)
        all_confidences.extend(probabilities)
        
        total += labels.size(0)
        correct += (predicted == labels.cpu().numpy()).sum()
        
        class_num = targets.item()
        stats[class_num]['total'] += 1
        if predicted == labels.cpu().numpy():
            stats[class_num]['correct'] += 1
        else:
            stats[class_num]['incorrect'] += 1

    accuracy = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds, average='binary', zero_division=1)
    precision = precision_score(all_labels, all_preds, average='binary', zero_division=1)
    avg_confidence = np.mean(all_confidences)

    print(f'Total images: {total}')
    print(f'Correct predictions: {correct}')
    print(f'Incorrect predictions: {total - correct}')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Average Confidence: {avg_confidence:.4f}')

    for class_num, stat in stats.items():
        class_accuracy = stat['correct'] / stat['total'] if stat['total'] > 0 else 0
        print(f'Class {class_num}: Total: {stat["total"]}, Correct: {stat["correct"]}, Incorrect: {stat["incorrect"]}, Accuracy: {class_accuracy:.4f}')

    return accuracy, recall, precision, avg_confidence

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = r'C:\Users\16074\Desktop\tiger\runs\run22\best_val_loss_model.pth'
    image_folder = r'C:\Users\16074\Desktop\dataset_tiger'

    model = load_model(model_path, device)
    accuracy, recall, precision, avg_confidence = predict_images(model, image_folder, device)
    print(f'Final Results - Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}, Average Confidence: {avg_confidence:.4f}')

if __name__ == '__main__':
    main()
