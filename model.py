import torch
import torchvision
import torch.nn as nn
import numpy as np
import pandas as pd
from collections import OrderedDict

from torchvision.models import ResNet
from torchvision.models.resnet import Bottleneck

model_type = 'resnext50' # or 'resnet50'
device = 'cuda:5'

# inheritance to add deep feature vector functionality to model
class resnet_model_with_dv(ResNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dv_prob = 0
        self.dv_weight = 0

    def forward(self, x, df=None):
        return model_forward(self, x, df)  
    
# import model checkpoint on ResNet-50 + convert to PyTorch
ckpt = 'ovis-%s-bal02/epoch_50.pth' % 'resnext50-5e-5' if model_type == 'resnext50' else 'resnet50-5e-5'

def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return resnet_model_with_dv(block=Bottleneck, layers=[3, 4, 6, 3], **kwargs)

def resnet50(pretrained=False, progress=True, **kwargs):
    return resnet_model_with_dv(block=Bottleneck, layers=[3, 4, 6, 3], **kwargs)
    
model = resnext50_32x4d() if model_type == 'resnext50' else torchvision.models.resnet50

model.fc = nn.Linear(2048, 23)
import_dict = torch.load(ckpt)['state_dict']
state_dict = OrderedDict()

# for purposes of getting intermediate layer outputs
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

# for purposes of conversion to PyTorch
for k, v in import_dict.items():
    if k.startswith('backbone'):
        state_dict[k.replace('backbone.', '')] = v
    elif k.startswith('head'):
        state_dict[k.replace('head.', '')] = v
    else:
        state_dict[k] = v

# load in converted checkpoint
#[print(i, k) for i, (k, v) in enumerate(model.named_parameters())]
model.load_state_dict(state_dict=state_dict)
print('Imported %s model from checkpoint: %s' % (model_type, ckpt))

#print([k for k,v in model.state_dict().items()])

'''
Supplementary functions to pull certain parts of networks in and out.
'''

def model_forward(self, x, df):
    # See note [TorchScript super()]
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    if df is not None:
        if np.random.random() < self.dv_prob:
            x += self.dv_weight * df
    x = torch.flatten(x, 1)
    x = self.fc(x)

    return x

@torch.no_grad()
def backbone(x):
    model.zero_grad()
    hk = model.avgpool.register_forward_hook(get_activation('deep_feat'))
    out = model(x)
    act = activation['deep_feat']
    hk.remove()
    return act

#model = model.to(device)
#sample_tensor = torch.rand([1,3,224,224]).to(device)'''