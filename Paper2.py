import torch

import torch.nn as nn

import torch.nn.functional as F



class CustomCNN(nn.Module):

    def __init__(self):

        super(CustomCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.fc1 = nn.Linear(256 * 2 * 2, 512) 
        self.fc2 = nn.Linear(512, 28) 

    def forward(self, x):
        x = x.to(next(self.parameters()).device)
        
        x = F.relu(self.conv1(x))
        x = self.pool(x) #16

        x = F.relu(self.conv2(x))
        x = self.pool(x) #8

        x = F.relu(self.conv3(x))
        x = self.pool(x)#4

        x = F.relu(self.conv4(x))
        x = self.pool(x)#2

        x = x.view(-1, 256 * 2 * 2)  

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
