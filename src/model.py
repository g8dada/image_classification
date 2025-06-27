import torch.nn as nn
from .config import CONFIG

class CNNModel(nn.Module):
    def __init__(self):                         # batch = 32
        super(CNNModel, self).__init__()
        self.conv_layer = nn.Sequential(        # shape: (32, 3, 150, 150)
            nn.Conv2d(3, 16, 3, padding=1),     # shape: (32, 16, 150, 150)
            nn.ReLU(),
            nn.MaxPool2d(2),                    # shape: (32, 16, 150, 150)
            nn.Conv2d(16, 32, 3, padding=1),    # shape: (32, 32, 150, 150)
            nn.ReLU(),
            nn.MaxPool2d(2)                     # shape: (32, 32, 75, 75)
        )
        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * (CONFIG["image_size"] // 4) * (CONFIG["image_size"] // 4), 128),         # 4: 2 times of Pooling, size -> 128
            nn.ReLU(),
            nn.Linear(128, CONFIG["num_classes"])                                                   # 128 -> 6 classes
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.fc_layer(x)
        return x
