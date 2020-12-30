import torch 
import torch.nn as nn
import torchvision.models as models

from pytorch_pretrained_vit import ViT

def freeze_layers(model):
    for param in model.parameters():
        param.requires_grad = False

class ShallowNet(nn.Module):
    def __init__(self, num_classes, fc_neurons=512):
        # input size: batch_sizex3x224x224
        super(ShallowNet, self).__init__()
        self.num_classes = num_classes
        self.fc_neurons = fc_neurons
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # output size from conv2d: batch_sizex16x220x220
        # output size from maxpool2d: batch_sizex16x110x110
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # output size from conv2d: batch_sizex32x106x106
        # output size from maxpool2d: batch_sizex32x53x53
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(53*53*32, self.fc_neurons),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(self.fc_neurons, self.num_classes)
            ) 
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.fc(out)
        return out

class ResNet(nn.Module):
    def __init__(self, num_classes, args):
        super(ResNet, self).__init__()
        self.num_classes = num_classes
        if args.model_type == 'resnet18':
            base_model = models.resnet18(pretrained=args.pretrained, progress=True)
        elif args.model_type == 'resnet152':
            base_model = models.resnet152(pretrained=args.pretrained, progress=True)
        self.model = base_model

        # Initialize/freeze weights
        if args.pretrained:
            freeze_layers(self.model)
        else:
            self.init_weights()
        
        # Classifier head
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    @torch.no_grad()
    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)
            
        self.apply(_init)
        nn.init.constant_(self.model.fc.weight, 0)
        nn.init.constant_(self.model.fc.bias, 0)
        
    def forward(self, x):
        out = self.model(x)
        return out

class VisionTransformer(nn.Module):
    def __init__(self, num_classes, args):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        base_model = ViT(name=args.model_type,
        pretrained = args.pretrained, 
        num_classes = self.num_classes,
        image_size = args.image_size)
        self.model = base_model
        
    def forward(self, x):
        out = self.model(x)
        return out
