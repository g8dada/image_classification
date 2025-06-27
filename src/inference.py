import torch
from torchvision import transforms
from PIL import Image
from src.model import CNNModel  # use your actual model class
import os

# 1. Load model
def load_model(model_path, device='cpu'):
    model = CNNModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# 2. Preprocess image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # add batch dim [1, 3, 64, 64]

# 3. Predict
def predict(model, image_tensor, class_names=None):
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = outputs.max(1)
        class_idx = predicted.item()
        return class_names[class_idx] if class_names else class_idx