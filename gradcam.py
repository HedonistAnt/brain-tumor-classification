from torchvision.models import resnet50
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import model
from dataset import get_dataloaders
from torchvision import datasets, transforms
import torch
from model import get_model
import torch.nn.functional as F
from pathlib import Path
class PadToSquare:
    def __call__(self, img):
        w, h = img.size
        max_dim = max(w, h)
        pad_left = (max_dim - w) // 2
        pad_top = (max_dim - h) // 2
        pad_right = max_dim - w - pad_left
        pad_bottom = max_dim - h - pad_top
        padding = (pad_left, pad_top, pad_right, pad_bottom)
        return T.functional.pad(img, padding, fill=0)

def load_model(model_path, num_classes=4, device=None, grayscale_input=True):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model directly without downloading
    model = get_model(num_classes=num_classes, grayscale_input=grayscale_input)
    model = model.to(device)

    # Load the weights without triggering gdown
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    for name, module in model.named_modules():
         print(name)

    return model, device

if __name__ == "__main__":
# Load your model (assume it's already loaded)
    model_path = r"D:\python_projects\brain-tumor-detection\trained_models\20250522_15_11_10_brain_tumor_model_epoch_12.pth"  # Update with your model filename
    _, _, test_loader, class_names = get_dataloaders()

    model, device = load_model(model_path, num_classes=len(class_names), grayscale_input=True)

    # Hook Grad-CAM to the last layer
    cam_extractor = GradCAM(model, target_layer='backbone.layer4.2.conv3')

    # Forward pass an image
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # convert to single-channel
        PadToSquare(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # normalize for grayscale
    ])

    image_path = r"D:\python_projects\brain-tumor-detection\data\Brain_Cancer\Testing\notumor\Te-no_0111.jpg"  
    true_label_name = Path(image_path).parent.name
    image = Image.open(image_path).convert("L") 

    input_tensor = transform(image).unsqueeze(0) # Add batch dimension and move to device
    out = model(input_tensor)
   


    # Choose top class
    class_idx = out.squeeze().argmax().item()
    class_name = class_names[class_idx]
    print(f"Predicted class: {class_name} (index: {class_idx})")
    probs = F.softmax(out, dim=1)
    confidence = probs[0, class_idx].item() * 100  
    # Extract cam
    activation_map = cam_extractor(class_idx, out)
    img = to_pil_image(input_tensor.squeeze(0))
    img_rgb = img.convert("RGB")
    # Overlay heatmap on original image
    result = overlay_mask(img_rgb, to_pil_image(activation_map[0], mode='F'), alpha=0.5)
    
    # Show result
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title(f"True class: {true_label_name}")
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(result)
    plt.title(f"Predicted: {class_name} ({confidence:.2f}%)")   
    plt.axis('off')
    plt.show()