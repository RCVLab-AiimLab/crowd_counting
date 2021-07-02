import torch.nn as nn
import torch
from torchvision import models
from utils import save_net,load_net
from caps import *
from caps2 import *
import torch.nn.functional as F


class CSRNet(nn.Module):
    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()
        self.seen = 0
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.avg = nn.AvgPool2d(kernel_size=28)
        
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(512, 512)

        self.comp_feat = [32,1]
        self.frontend = make_layers(self.frontend_feat)
        self.caps = CapsNet(3)
        self.comp_mod = make_layers_comp(self.comp_feat)
        # self.caps_att = make_layers_caps_att()
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        self.network = CapsuleNetwork(image_width=28,
                         image_height=28,
                         image_channels=512,
                         conv_inputs=conv_inputs,
                         conv_outputs=conv_outputs,
                         num_primary_units=num_primary_units,
                         primary_unit_size=primary_unit_size,
                         num_output_units=10, # one for each MNIST digit
                         output_unit_size=output_unit_size).cuda()
        if not load_weights:
            mod = models.vgg16(pretrained = True)
            self._initialize_weights()
            for i in range(len(self.frontend.state_dict().items())):
                list(self.frontend.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[i][1].data[:]
    def forward(self,x):
        x = self.frontend(x)
        # x_att = self.conv_att(x).unsqueeze(-1).unsqueeze(-1)
        # x_att = self.avg(x)


        x_att = F.adaptive_avg_pool2d(x, (1, 1))
        x_att = self.flatten(x_att)
        x_att = self.linear(x_att)
        x_att = F.relu(x_att)
        x_att = self.linear(x_att)
        x_att = F.tanh(x_att)
        x_att = x_att.unsqueeze(-1).unsqueeze(-1)
        x_att = x_att * x
        x_att = x_att + x


        # print(x.shape)
        # x_att = self.caps(x_att)[0]

        x_att = self.network(x_att)
        # print(x_att.shape)
        
        # print(x_att.shape)
        # x_caps_att = x_conv_att*x_att[0]
        # x_conv_att = x_conv_att.view((1,x_conv_att.shape[3],x_conv_att.shape[1],x_conv_att.shape[2]))
        # x_caps_att = x_caps_att.reshape((1,x_caps_att.shape[3],x_caps_att.shape[1],x_caps_att.shape[2]))
        # x_att = torch.cat((x_conv_att,x_caps_att),dim=1)

        x_att = self.comp_mod(x_att)
        return x_att
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            
                
def make_layers(cfg, in_channels = 3,batch_norm=False,dilation = False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)                 

def make_layers_conv_att(cfg, in_channels = 3, batch_norm=False, dilation = False):
    layers = []
    for v in cfg:
        if v == 'A':
            layers += [nn.AvgPool2d(kernel_size=28), nn.Flatten()]
        else:
            layers += [nn.Linear(in_features = 512, out_features = 512)]
            in_channels = v
    return nn.Sequential(*layers)  

def make_layers_comp(cfg, in_channels = 64,batch_norm=True,dilation = False):
    layers = []
    for v in cfg:
        print(v)
        print(in_channels)
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv1 = nn.Conv2d(in_channels, v, kernel_size=1)
            # conv2 = nn.Conv2d(32, v, kernel_size=1)
            layers += [conv1, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)  


            

            