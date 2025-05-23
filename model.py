import torch
import torch.nn as nn

class RadImageNetResNet50Classifier(nn.Module):
    def __init__(self, num_classes=4, grayscale_input=True):
        super().__init__()

        # Load pretrained feature extractor
        self.backbone = torch.hub.load("Warvito/radimagenet-models", "radimagenet_resnet50", trust_repo=True)

        if grayscale_input:
            self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Add classifier layers
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.backbone(x)              # shape: [B, 2048, H, W]
        x = self.pool(x)                  # shape: [B, 2048, 1, 1]
        x = self.flatten(x)              # shape: [B, 2048]
        x = self.classifier(x)           # shape: [B, 3]
        return x

def get_model(num_classes=4, grayscale_input=True):
    return RadImageNetResNet50Classifier(num_classes=num_classes, grayscale_input=grayscale_input)
