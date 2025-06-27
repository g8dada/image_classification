import torch
import torch.nn as nn
import torch.optim as optim
from .dataset import get_dataloaders
from .model import CNNModel
from .config import CONFIG
from .utils import evaluate
import os
from tqdm import tqdm

def save_checkpoint(model, epoch, path=CONFIG["checkpoint_dir"]):
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(path, f"model_epoch_{epoch+1}.pt"))


def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNModel().to(device)
    train_loader, val_loader = get_dataloaders()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

    for epoch in range(CONFIG["epochs"]):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc="Training"):
            images, labels = images.to(device), labels.to(device)
            
            # print(f'image: \n{images}\n\n')
            # print(f'shape: {images.shape}\n\n\n')   # shape: torch.Size([32, 3, 150, 150])
            # print(f'label: \n{labels}')             # shape: torch.Size([32])
            # input()

            optimizer.zero_grad()
            outputs = model(images)
            
            # print(f'ouputs: {outputs}\n\n')                # shape: (32, 6)
            # print(f'ouput 1: {outputs[0]}\n\n')            # ouput 1: tensor([-0.0211, -0.0611, -0.0670,  0.1049,  0.0111,  0.1013], device='cuda:0', grad_fn=<SelectBackward0>)
            # print(f'ouput size: {outputs[0].shape}\n\n')   # ouput size: torch.Size([6])
            # input()
            
            loss = criterion(outputs, labels)
            loss.backward()                    # calculate gradients
            optimizer.step()                   # update parameters

            running_loss += loss.item()
            
            # print(outputs.max(1))              # outputs max value and corresponding indices along 1st dimension
            _, predicted = outputs.max(1)
            # print(predicted)
            # input()
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            # break

        if (epoch+1) % 5 == 0:
            save_checkpoint(model, epoch, path="checkpoints")
        print(f"Epoch {epoch+1}/{CONFIG['epochs']}, Loss: {running_loss:.4f}, Accuracy: {100 * correct / total:.2f}%")
    print("Done training!\n")
    
    # Evaluation
    val_loss, val_acc = evaluate(model, val_loader, criterion, device='cuda')
    print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc*100:.2f}%")
    
