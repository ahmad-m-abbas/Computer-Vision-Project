import torch
import torch.nn as nn
import torch.nn.functional as F

class ArabicCharCNN(nn.Module):
    def __init__(self):
        super(ArabicCharCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(in_features=256*4*4, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=1024)
        self.fc3 = nn.Linear(in_features=1024, out_features=512)
        self.fc4 = nn.Linear(in_features=512, out_features=28)  

        self.dropout = nn.Dropout(p=0.8)

    def forward(self, x):
        x = x.to(next(self.parameters()).device)
        
        x = self.pool(F.relu(self.bn1(self.conv1(x)))) #16
        x = self.pool(F.relu(self.bn2(self.conv2(x)))) #8
        x = self.pool(F.relu(self.bn3(self.conv3(x)))) #4
        # print(x.shape)

        x = x.view(-1, 256 * 4 * 4)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)

        return x

