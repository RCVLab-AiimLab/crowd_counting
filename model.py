import torch
import torch.nn as nn
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

import numpy as np 
from copy import deepcopy
from torchvision import models

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 128, layers[0])
        self.layer2 = self._make_layer(block, 256, layers[1], stride=2, dilate=replace_stride_with_dilation[0])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride), norm_layer(planes * block.expansion),)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


class MSPSNet(nn.Module):
    def __init__(self, load_weights=False, verbose=False, backend='vgg'):
        super(MSPSNet, self).__init__()

        if backend == 'resnet':
            self.backend = _resnet('wide_resnet101_2', Bottleneck, [3, 4, 6, 3], pretrained=False, progress=True)
        elif backend == 'vgg':
            self.backend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
            self.backend_feat_1 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256]
            self.backend_feat_2 = [64, 64, 'M', 128, 128]
            self.backend = make_layers(self.backend_feat, in_channels=3)
            self.backend_1 = make_layers(self.backend_feat_1, in_channels=3)
            self.backend_2 = make_layers(self.backend_feat_2, in_channels=3)    
        
        self.frontend_feat  = [512, 512, 512, 256, 128, 64]
        self.frontend_feat_1  = [256, 256, 256, 256, 128, 128, 128, 64]
        self.frontend_feat_2  = [128, 128, 128, 64, 64]
        #self.frontend_feat_2  = [128, 128, 128, 128, 128, 64, 64, 64, 64, 64]
        self.frontend = make_layers(self.frontend_feat, in_channels=512, dilation=True)
        self.frontend_1 = make_layers(self.frontend_feat_1, in_channels=256, dilation=True)
        self.frontend_2 = make_layers(self.frontend_feat_2, in_channels=128, dilation=True)

        self.output_layer_0 = OutConv(64, 1)
        self.output_layer_1 = OutConv(64, 1)
        self.output_layer_2 = OutConv(64, 1)
        self.output_layer_3 = OutConv(192, 1)
        
        self.down_fuse_2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=2, stride=2), nn.LeakyReLU(inplace=True),
                                        nn.Conv2d(64, 64, kernel_size=2, stride=2), nn.LeakyReLU(inplace=True),
                                        nn.Conv2d(64, 64, kernel_size=2, stride=2), nn.LeakyReLU(inplace=True),)

        self.down_fuse_1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=2, stride=2), nn.LeakyReLU(inplace=True),
                                        nn.Conv2d(64, 64, kernel_size=2, stride=2), nn.LeakyReLU(inplace=True),)

        self.down_fuse_0 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=2, stride=2), nn.LeakyReLU(inplace=True),)

        if backend == 'resnet':
            if not load_weights:
                mod = models.wide_resnet50_2(pretrained=True)
                self._initialize_weights()
                for i in range(len(self.backend.state_dict().items())):
                    temp = list(self.backend.state_dict().items())[i]
                    if len(temp[1].size()) > 0:
                        list(self.backend.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[i][1].data[:]
        
        elif backend == 'vgg':
            if not load_weights:
                mod = models.vgg16(pretrained=True)
                self._initialize_weights()
                for i in range(len(self.backend.state_dict().items())):
                    list(self.backend.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[i][1].data[:]
                for i in range(len(self.backend_1.state_dict().items())):
                    list(self.backend_1.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[i][1].data[:]
                for i in range(len(self.backend_2.state_dict().items())):
                    list(self.backend_2.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[i][1].data[:]


        self.info(verbose=verbose)


    def forward(self, x_im, x_depth=None, training=True):
        device = x_im.device
        x = self.backend(x_im)

        x0_64 = self.frontend(x)
        x0 = self.output_layer_0(x0_64)
        x0 = x0.squeeze(1)  # count_0

        x1 = self.backend_1(x_im)
        x1_64 = self.frontend_1(x1)
        x1 = self.output_layer_1(x1_64)
        x1 = x1.squeeze(1)  # count_1

        x2 = self.backend_2(x_im)
        x2_64 = self.frontend_2(x2)
        x2 = self.output_layer_2(x2_64)
        x2 = x2.squeeze(1)  # count_2

        x00 = self.down_fuse_0(x0_64)
        x11 = self.down_fuse_1(x1_64)
        x22 = self.down_fuse_2(x2_64)
        
        x_fuse = torch.cat([x00, x11, x22], dim=1)
        x_fuse = self.output_layer_3(x_fuse)
        x_fuse = x_fuse.squeeze(1)  # count_fuse


        return x0, x1, x2, x_fuse


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            
                
    def info(self, verbose=False):  # print model information
        model_info(self, verbose)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, dilation=False):
        super().__init__()

        if dilation:
            d_rate = 2
        else:
            d_rate = 1

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, d_rate)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels//2, out_channels, d_rate)


    def forward(self, x1):
        x1 = self.up(x1)

        return self.conv(x1)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, d_rate=1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=d_rate, dilation=d_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=d_rate, dilation=d_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=d_rate, dilation=d_rate),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.LeakyReLU(inplace=True)]
            else:
                layers += [conv2d, nn.LeakyReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)  


def model_info(model, verbose=False, img_size=128):
    # Model information. 
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    if verbose:
        print('%5s %40s %9s %12s %20s %10s %10s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    
    fs = ''

    print(f"Model Summary: {len(list(model.modules()))} layers, {n_p} parameters, {n_g} gradients{fs}")


class ComputeLoss:
    # Compute losses
    def __init__(self, model):
        super(ComputeLoss, self).__init__()
        device = next(model.parameters()).device  # get model device

        # Define criteria
        self.MSELoss = nn.MSELoss(reduction='sum')
        self.BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1], device=device))


    def __call__(self, p0, p1, p2, p3, targets): 
        device = targets.device
        
        # Losses
        lcount_0 = torch.zeros(1, device=device)
        lcount_1 = torch.zeros(1, device=device)
        lcount_2 = torch.zeros(1, device=device)
        lcount_3 = torch.zeros(1, device=device)
        
        tcount_0 = torch.zeros_like(p0, device=device)  
        tcount_1 = torch.zeros_like(p1, device=device)  
        tcount_2 = torch.zeros_like(p2, device=device)     
        tcount_t = torch.zeros_like(p3, device=device)     

        indices_t, indices_0, indices_1, indices_2 = self.build_targets(targets, p2.size(1)*2, p2.size(2)*2) 
        
        gj, gi = indices_2[0]  
        
        b = 0
        n = gi.shape[0]  
        if n:

            indices_t = indices_t[0]
            indices_t = torch.stack(indices_t)
            nonempty, count = torch.unique(indices_t, dim=1, return_counts=True) 

            for k in range(nonempty.size(1)):
                indx = nonempty[:, k]
                tcount_t[b, indx[1], indx[0]] = float(count[k])

            indices_0 = indices_0[0]
            indices_0 = torch.stack(indices_0)
            nonempty, count = torch.unique(indices_0, dim=1, return_counts=True) 

            for k in range(nonempty.size(1)):
                indx = nonempty[:, k]
                tcount_0[b, indx[1], indx[0]] = float(count[k])
            
            indices_1 = indices_1[0]
            indices_1 = torch.stack(indices_1)
            nonempty, count = torch.unique(indices_1, dim=1, return_counts=True) 

            for k in range(nonempty.size(1)):
                indx = nonempty[:, k]
                tcount_1[b, indx[1], indx[0]] = float(count[k])

            tcount_2[b, gi, gj] = 1 

            '''indices_2 = indices_2[0]
            indices_2 = torch.stack(indices_2)
            nonempty, count = torch.unique(indices_2, dim=1, return_counts=True) 

            for k in range(nonempty.size(1)):
                indx = nonempty[:, k]
                tcount_2[b, indx[1], indx[0]] = float(count[k])'''

        H0 = 0.1
        H1 = 0.2
        H2 = 0.3
        H3 = 0.1

        lcount_0 += (self.MSELoss(p0, tcount_0) * H0) 
        lcount_1 += (self.MSELoss(p1, tcount_1) * H1) 
        lcount_2 += (self.MSELoss(p2, tcount_2) * H2)  
        lcount_3 += (self.MSELoss(p3, tcount_t) * H3)  
        
        return  [lcount_0, lcount_1, lcount_2, lcount_3], lcount_0.detach(), lcount_1.detach(), lcount_2.detach(), lcount_3.detach()


    def build_targets(self, targets, s0, s1):
        # Build targets for compute_loss()
        device = targets.device
        
        gain = torch.ones(2, device=device)  # normalized to gridspace gain

        ##### count
        indices_t = []
        gain = torch.tensor((s0//16, s1//16), device=device)  
        t = targets * gain

        gij = t.long()
        gi, gj = gij.T  # grid xy indices

        indices_t.append((gj, gi))  

        ##### count
        indices_0 = []
        gain = torch.tensor((s0//8, s1//8), device=device) 
        t = targets * gain

        gij = t.long()
        gi, gj = gij.T  # grid xy indices

        indices_0.append((gj, gi))  

        ##### count
        indices_1 = []
        gain = torch.tensor((s0//4, s1//4), device=device) 
        t = targets * gain

        gij = t.long()
        gi, gj = gij.T  # grid xy indices

        indices_1.append((gj, gi))  
        
        ##### loc
        indices_2 = []
        gain = torch.tensor((s0//2, s1//2), device=device)
        t = targets * gain

        gij = t.long()
        gi, gj = gij.T  # grid xy indices

        indices_2.append((gj, gi))  

        return indices_t, indices_0, indices_1, indices_2
