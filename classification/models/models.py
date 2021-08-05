import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import einops
from einops.layers.torch import Rearrange

from pytorch_pretrained_vit import ViT, ViTConfigExtended, PRETRAINED_CONFIGS

def model_selection(args, device):
    # initiates model and loss     
    if args.model_name=='shallow':
        model = ShallowNet(args)
    elif args.model_name=='resnet18' or args.model_name=='resnet152':
        model = ResNet(args)
    else:
        model = VisionTransformer(args)

    if args.checkpoint_path:
            if args.load_partial_mode:
                model.model.load_partial(weights_path=args.checkpoint_path, 
                pretrained_image_size=self.configuration.pretrained_image_size, 
                pretrained_mode=args.load_partial_mode, verbose=True)
            else:
                state_dict = torch.load(args.checkpoint_path, map_location=torch.device('cpu'))
                #state_dict = torch.load(args.checkpoint_path)
                expected_missing_keys = []
                if args.transfer_learning:
                    # Modifications to load partial state dict
                    '''
                    if ('patch_embedding.weight' in state_dict and self.configuration.num_channels != self.configuration.pretrained_num_channels):
                        expected_missing_keys += ['model.patch_embedding.weight', 'model.patch_embedding.bias']
                    if ('pre_logits.weight' in state_dict and load_repr_layer==False):
                        expected_missing_keys += ['model.pre_logits.weight', 'model.pre_logits.bias']
                    '''
                    if ('model.fc.weight' in state_dict):
                        expected_missing_keys += ['model.fc.weight', 'model.fc.bias']
                    for key in expected_missing_keys:
                        state_dict.pop(key)
                        #print(key)
                ret = model.load_state_dict(state_dict, strict=False)
                print('''Missing keys when loading pretrained weights: {}
                    Expected missing keys: {}'''.format(ret.missing_keys, expected_missing_keys))
                print('Unexpected keys when loading pretrained weights: {}'.format(ret.unexpected_keys))
                print('Loaded from custom checkpoint.')

    model.to(device)

    return model
    

def freeze_layers(model):
    for param in model.parameters():
        param.requires_grad = False


class ShallowNet(nn.Module):
    def __init__(self, args, fc_neurons=512):
        # input size: batch_sizex3x224x224
        super(ShallowNet, self).__init__()
        self.num_classes = args.num_classes
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
    def __init__(self, args):
        super(ResNet, self).__init__()
        self.num_classes = args.num_classes
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
    def __init__(self, args):
        super(VisionTransformer, self).__init__()
        
        def_config = PRETRAINED_CONFIGS['{}'.format(args.model_name)]['config']
        self.configuration = ViTConfigExtended(**def_config)
        self.configuration.num_classes = args.num_classes
        self.configuration.image_size = args.image_size
        
        if hasattr(args, 'vis_attention'):
            base_model = ViT(self.configuration, name=args.model_name, 
            pretrained=args.pretrained, ret_attn_scores=True)
        else:
            base_model = ViT(self.configuration, name=args.model_name, pretrained=args.pretrained, 
            load_fc_layer=not(args.interm_features_fc), ret_interm_repr=args.interm_features_fc)
        self.model = base_model

        if args.interm_features_fc:
            self.class_head = nn.Sequential(
                        nn.Linear(self.configuration.num_hidden_layers, 1),
                        Rearrange(' b d 1 -> b d'),
                        nn.ReLU(),
                        nn.LayerNorm(self.configuration.hidden_size, eps=self.configuration.layer_norm_eps),
                        nn.Linear(self.configuration.hidden_size, self.configuration.num_classes)
                    )
            if args.exclusion_loss:
                self.exclusion_loss = nn.KLDivLoss(reduction='batchmean')
                self.temperature = args.temperature if args.temperature else 1
                
    def forward(self, x):
        exclusion_loss = 0
        if hasattr(self, 'class_head'):
            x, interm_features = self.model(x)
        else:
            x = self.model(x)
        
        if hasattr(self, 'class_head'):
            if hasattr(self, 'exclusion_loss'):
                for i in range(len(interm_features) - 1):
                    exclusion_loss += self.exclusion_loss(
                        F.log_softmax(interm_features[:, i, 0]/self.temperature, dim=1), 
                        F.softmax(interm_features[:, i+1, 0]/self.temperature, dim=1)
                        ) *
            interm_features = torch.stack(interm_features, dim=-1)
            x = self.class_head(interm_features[:, 0])
        
        if exclusion_loss:
            return x, exclusion_loss
        return x