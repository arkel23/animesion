import torch 
import torch.nn as nn

class simpleNet(nn.Module):
    def __init__(self, num_classes, fc_neurons=512):
        # input size: batch_sizex3x128x128
        super(simpleNet, self).__init__()
        self.num_classes = num_classes
        self.fc_neurons = fc_neurons
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # output size from conv2d: batch_sizex16x124x124
        # output size from maxpool2d: batch_sizex16x62x62
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # output size from conv2d: batch_sizex32x58x58
        # output size from maxpool2d: batch_sizex32x29x29
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(29*29*32, self.fc_neurons),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(self.fc_neurons, self.num_classes)
            ) 
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.fc(out)
        return out