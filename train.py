import torch
from dataset import get_dataloaders
import torch.nn as nn
import torch.optim as optim
from model import get_model
from datetime import datetime
from pathlib import Path
import os, shutil
import numpy as np
import json

try:
    from google.colab import drive
except ImportError:
    drive = None  # Not running in Google Colab


train_losses = []
val_accuracies = []
class BrainTumorTrainer:
    def __init__(self, num_classes=4, lr=1e-4, device=None,train_ratio=0.9, val_ratio=0.1):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Data
        self.train_loader, self.val_loader, self.test_loader, self.classes = get_dataloaders(train_ratio=train_ratio, val_ratio=val_ratio)
        self.num_classes = num_classes

        # Model
        self.model = get_model(num_classes=num_classes).to(self.device)

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self, epochs=10):
       
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0

            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            print(f"🟢 Epoch {epoch + 1} | Training Loss: {running_loss:.4f}")
            train_losses.append(running_loss)
            self.validate(epoch)

    def validate(self, epoch=None):
        self.model.eval()
        correct, total = 0, 0
        if epoch == "Test": 
            loader = self.test_loader
        else:
            loader = self.val_loader

        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = 100 * correct / total
        if isinstance(epoch, int):
            tag = f"Epoch {epoch + 1}"
            val_accuracies.append(acc)
            print(f"🔵 {tag} Validation Accuracy: {acc:.2f}%")
            timestamp = datetime.now().strftime("%Y%m%d_%H_%M_%S")
            torch.save(self.model.state_dict(), Path(__file__).resolve().parent / "trained_models" / f"{timestamp}_brain_tumor_model_epoch_{epoch+1}.pth")
        elif epoch == "Test":
            print(f"🔵 Test Accuracy: {acc:.2f}%")
        
    def test(self):
        self.validate(epoch="Test")
    
if __name__ == "__main__":

    trainer = BrainTumorTrainer()
    trainer.train(epochs=10)
    trainer.test()
    with open(os.path.join(Path(__file__).resolve().parent / "trained_models", "train_log.json"), "w") as f:
            json.dump({
                "train_losses": train_losses,
                "val_accuracies": val_accuracies
            }, f)
    if drive:
        source_folder = Path(__file__).resolve().parent / "trained_models" 
        destination_folder = "/content/drive/MyDrive/brain-tumor-checkpoints/trained_models"
        os.makedirs(os.path.dirname(destination_folder), exist_ok=True)
        shutil.copytree(source_folder, destination_folder, dirs_exist_ok=True)
        print(f"✅ Folder saved to Google Drive: {destination_folder}")
    
        

    
