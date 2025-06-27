import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from .config import CONFIG

def get_dataloaders():
    transform = transforms.Compose([
        transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(os.path.join(CONFIG["data_dir"], "seg_train"), transform=transform)
    val_dataset = datasets.ImageFolder(os.path.join(CONFIG["data_dir"], "seg_test"), transform=transform)

    # print(train_dataset.classes)        # ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
    # print(train_dataset.samples)        # e.g. ('data/seg_train/street/999.jpg', 5) <- a tuple
    # input()
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

    return train_loader, val_loader
