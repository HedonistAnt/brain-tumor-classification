import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path



def get_dataloaders(data_dir=Path(__file__).resolve().parent / "data"/"Brain_Cancer", batch_size=32, img_size=224, train_ratio=0.9, val_ratio=0.1):

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # convert to single-channel
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # normalize for grayscale
    ])

    training_dataset = datasets.ImageFolder(root=data_dir/"Training", transform=transform)
    testing_dataset = datasets.ImageFolder(root=data_dir/"Testing", transform=transform)
    # Print class mapping
    class_names = training_dataset.classes
    print(f"✅ Found classes: {class_names}")

    total_size = len(training_dataset)
    train_size = int(train_ratio * total_size)
    val_size = total_size - train_size
    

    training_dataset, val_dataset  = torch.utils.data.random_split(training_dataset, [train_size, val_size])

    train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(testing_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, class_names