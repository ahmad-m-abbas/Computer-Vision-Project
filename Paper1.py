import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN14(nn.Module):
    def __init__(self):
        super(CNN14, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1) # (H -K + 2P)/S + 1, (32-3+2)/1 + 1 = 32
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        # Max pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(512 * 2 * 2, 4096)  # Adjust the input features to match your input image size
        self.fc2 = nn.Linear(4096, 28)  # Output features should match the number of classes

        # Dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.to(next(self.parameters()).device)
        # First block
        x = F.relu(self.conv1(x)) # 32
        x = F.relu(self.conv2(x)) # 32
        x = self.pool(x) # 16

        # Second block
        x = F.relu(self.conv3(x)) #16
        x = F.relu(self.conv4(x)) #16
        x = self.pool(x) #8

        # Third block
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool(x) #4

        # Fourth block
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = self.pool(x) # 2

        # Flatten the output for the dense layer
        x = x.view(-1, 512 * 2 * 2)  # Adjust the features to match your input image size

        # Dense layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
