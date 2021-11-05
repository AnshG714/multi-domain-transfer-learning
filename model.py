import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

class Net2(nn.Module):

    def __init__(self):
        super(Net2, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(44944, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 200)
        self.fc3 = nn.Linear(200, 200)
        self.fc4 = nn.Linear(200, 196)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.fc = nn.Sequential(
            nn.Linear(512, 1000),
            nn.Linear(1000, 1000),
            nn.Linear(1000, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)