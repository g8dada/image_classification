import os
import torch
from tqdm import tqdm

def save_checkpoint(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def evaluate(model, dataloader, criterion, device='cpu'):
    model.eval()  # Important: turn off dropout, batchnorm, etc.
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():  # no gradients needed during evaluation
        for images, labels in tqdm(dataloader, desc='Evaluating'):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)  # multiply by batch size
            _, predicted = outputs.max(1)
            total_correct += predicted.eq(labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return avg_loss, accuracy
