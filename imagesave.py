import os
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score


def create_run_folder(base_folder="runs"):
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    
    run_folders = [f for f in os.listdir(base_folder) if f.startswith("run")]
    run_number = len(run_folders) + 1
    run_folder = os.path.join(base_folder, f"run{run_number}")
    os.makedirs(run_folder)
    
    return run_folder

def save_plots(run_folder, metrics):
    epochs = range(1, len(metrics['train_loss']) + 1)
    
    # Plot training and validation loss
    plt.figure()
    plt.plot(epochs, metrics['train_loss'], label='Train Loss')
    plt.plot(epochs, metrics['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Train and Validation Loss')
    plt.savefig(os.path.join(run_folder, 'loss.png'))
    plt.close()
    
    # Plot training and validation accuracy
    plt.figure()
    plt.plot(epochs, metrics['train_acc'], label='Train Accuracy')
    plt.plot(epochs, metrics['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Train and Validation Accuracy')
    plt.savefig(os.path.join(run_folder, 'accuracy.png'))
    plt.close()

    # Plot precision
    plt.figure()
    plt.plot(epochs, metrics['train_precision'], label='Train Precision')
    plt.plot(epochs, metrics['val_precision'], label='Validation Precision')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.legend()
    plt.title('Train and Validation Precision')
    plt.savefig(os.path.join(run_folder, 'precision.png'))
    plt.close()

    # Plot recall
    plt.figure()
    plt.plot(epochs, metrics['train_recall'], label='Train Recall')
    plt.plot(epochs, metrics['val_recall'], label='Validation Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.legend()
    plt.title('Train and Validation Recall')
    plt.savefig(os.path.join(run_folder, 'recall.png'))
    plt.close()
