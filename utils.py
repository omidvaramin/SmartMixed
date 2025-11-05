import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

def save_loss_curves(train_losses, val_losses, output_dir, activation_name=None):
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    if activation_name:
        plt.title(f'Loss Curves ({activation_name})')
    else:
        plt.title('Loss Curves')
    plt.savefig(os.path.join(output_dir, 'loss_curves.png'), dpi=300)
    plt.close()

def save_confusion_matrix(y_true, y_pred, output_dir, class_names=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    return cm

def per_class_accuracy(y_true, y_pred, num_classes):
    accs = []
    for i in range(num_classes):
        idx = (y_true == i)
        if np.sum(idx) == 0:
            accs.append(0)
        else:
            accs.append(np.mean(y_pred[idx] == i))
    return accs

def save_per_class_accuracy(accs, output_dir):
    with open(os.path.join(output_dir, 'per_class_accuracy.txt'), 'w') as f:
        for i, acc in enumerate(accs):
            f.write(f'Class {i}: {acc:.4f}\n')

def save_total_accuracy(acc, output_dir):
    with open(os.path.join(output_dir, 'total_accuracy.txt'), 'w') as f:
        f.write(f'Total accuracy: {acc:.4f}\n')

def save_best_loss(loss, output_dir):
    with open(os.path.join(output_dir, 'best_test_loss.txt'), 'w') as f:
        f.write(f'Best test loss: {loss:.6f}\n')
