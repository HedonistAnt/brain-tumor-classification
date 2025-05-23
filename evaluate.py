import torch
from dataset import get_dataloaders
from model import get_model
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_model(model_path, num_classes=4, device=None, grayscale_input=True):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model directly without downloading
    model = get_model(num_classes=num_classes, grayscale_input=grayscale_input)
    model = model.to(device)

    # Load the weights without triggering gdown
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, device


def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel='True label',
        xlabel='Predicted label',
        title='Confusion Matrix'
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()


def confusion_matrix(preds, labels, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(labels, preds):
        cm[t, p] += 1
    return cm


def classification_report(cm, class_names):
    print("Classification Report:")
    print(f"{'Class':<15}{'Precision':<10}{'Recall':<10}{'F1-Score':<10}{'Support':<10}")
    for i, class_name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0
        support = cm[i, :].sum()
        print(f"{class_name:<15}{precision:<10.2f}{recall:<10.2f}{f1:<10.2f}{support:<10}")


def evaluate_model(model, loader, device, class_names):
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_preds, all_labels, len(class_names))  # FIXED ORDER
    plot_confusion_matrix(cm, class_names)
    classification_report(cm, class_names)


model_path = r"D:\python_projects\brain-tumor-detection\trained_models\20250522_15_11_10_brain_tumor_model_epoch_12.pth"  # Update with your model filename
_, _, test_loader, class_names = get_dataloaders(train_ratio=0.8, val_ratio=0.1)

model, device = load_model(model_path, num_classes=len(class_names), grayscale_input=True)
evaluate_model(model, test_loader, device, class_names)