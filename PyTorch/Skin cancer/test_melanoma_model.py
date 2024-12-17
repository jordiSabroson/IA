import os
import random
from torchvision import transforms
from PIL import Image
import torch
from torch import nn


# Carpetas con las imágenes
benign_dir = "data/test/benign"
malignant_dir = "data/test/malignant"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

class MelanomaDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=5,stride=2),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=2),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0),
            nn.Flatten(),
            nn.Linear(10816,100),
            nn.ReLU(),
            nn.Linear(100,2)
        )
    
    def forward(self, X):
        X = self.features(X)
        return X

def load_random_images(folder, num_images):
    all_images = os.listdir(folder)
    selected_images = random.sample(all_images, num_images)
    loaded_images = []
    for img_name in selected_images:
        img_path = os.path.join(folder, img_name)
        image = Image.open(img_path).convert('RGB')  # Convertir a RGB si no lo está
        loaded_images.append((img_name, transform(image)))
    return loaded_images

benign_images = load_random_images(benign_dir, 5)
malignant_images = load_random_images(malignant_dir, 5)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MelanomaDetector().to(device)
model.load_state_dict(torch.load("melanoma_model.pth", weights_only=False))
model.eval()

print("Benign")
for img_name, img_tensor in benign_images:
    img_tensor = img_tensor.unsqueeze(0).to(device) 
    output = model(img_tensor)
    _, predicted = torch.max(output, 1)
    class_name = "benign" if predicted.item() == 0 else "malignant"
    print(f"Imagen: {img_name}, Predicción: {class_name}")

print("Malignant")
for img_name, img_tensor in malignant_images:
    img_tensor = img_tensor.unsqueeze(0).to(device) 
    output = model(img_tensor)
    _, predicted = torch.max(output, 1)
    class_name = "benign" if predicted.item() == 0 else "malignant"
    print(f"Imagen: {img_name}, Predicción: {class_name}")
