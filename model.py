import torch.nn as nn
import torch
from torchvision import models
from utils import save_net,load_net
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

class CSRNet(nn.Module):
    def __init__(self, depth=None, load_weights=False, reconstruction_type='FC', 
                    imsize=64, num_classes=10, routing_iterations=3, primary_caps_gridsize=6, img_channel=3, 
                    batchnorm=False, loss='L2', num_primary_capsules=32, leaky_routing=False):
        
        super(CSRNet, self).__init__()
        self.seen = 0
        self.capsnet = self.make_layers(capsnet=capsnet, in_channels=self.frontend_feat[-1], primary_caps_gridsize=primary_caps_gridsize, imsize=imsize)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        self.use_reconstruction = True
        if not load_weights:
            mod = models.vgg16(pretrained=True)
            self._initialize_weights()
            for i in range(len(self.frontend.state_dict().items())):
                list(self.frontend.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[i][1].data[:]

    def forward(self, x, target=None):
        self.input_shape = x.shape[1:]
        #conv = self.frontend(x)
        conv = self.conv_layer(x)
        primary_capsules = self.primary_capsules(conv)
        digit_caps = self.digit_caps(primary_capsules)
        reconstruction, masked = self.decoder(digit_caps, target)

        return digit_caps, reconstruction, masked

        #return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layers(self, csrnet_cfg=None, in_channels=3, batch_norm=False, kernel_size=3, dilation=False, capsnet=False, imsize=64, 
                    img_channel=3, batchnorm=False, leaky_routing=False, primary_caps_gridsize=6,
                    num_primary_capsules=32, routing_iterations=3, reconstruction_type='FC',
                    loss='L2'):
        
        if not capsnet:
            if dilation:
                d_rate = 2
            else:
                d_rate = 1
            layers = []
            for v in csrnet_cfg:
                if v == 'M':
                    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                else:
                    conv2d = nn.Conv2d(in_channels, v, kernel_size=kernel_size, padding=d_rate, dilation=d_rate)
                    if batch_norm:
                        layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                    else:
                        layers += [conv2d, nn.ReLU(inplace=True)]
                    in_channels = v
            return nn.Sequential(*layers) 

        elif capsnet:  
            self.conv_layer = ConvLayer(in_channels=3, batchnorm=batchnorm)
            self.leaky_routing = leaky_routing
            self.primary_capsules = PrimaryCapsules(in_channels=in_channels, primary_caps_gridsize=primary_caps_gridsize, batchnorm=batchnorm, num_capsules=num_primary_capsules)
            #self.primary_capsules = PrimaryCapsules(input_shape=(256, 20, 20), capsule_dim=8, out_channels=32, kernel_size=9, stride=2)
            #self.routing = Routing(caps_dim_before=8, caps_dim_after=16, n_capsules_before=6*6*32, n_capsules_after=10, n_routing_iter=3)
            #self.norm = Norm()
            #self.decoder = Decoder(16, int(np.prod([3,32,32])))

            self.digit_caps = ClassCapsules(num_routes=num_primary_capsules*primary_caps_gridsize*primary_caps_gridsize, routing_iterations=routing_iterations, leaky=leaky_routing, in_channels=8)
            if reconstruction_type == "FC":
                self.decoder = ReconstructionModule(imsize=imsize, img_channel=3, batchnorm=batchnorm)
            elif reconstruction_type == "Conv32":
                self.decoder = SmallNorbConvReconstructionModule(imsize=imsize, img_channel=3, batchnorm=batchnorm)            
            else:
                self.decoder = ConvReconstructionModule(imsize=imsize, img_channel=3, batchnorm=batchnorm)
            
            if loss == "L2":
                self.reconstruction_criterion = nn.MSELoss(reduction="none")
            if loss == "L1":
                self.reconstruction_criterion = nn.L1Loss(reduction="none")
            

    def loss(self, images, labels, capsule_output,  reconstruction, alpha):
        marg_loss = 0 #self.margin_loss(capsule_output, labels)
        rec_loss = self.reconstruction_loss(images, reconstruction)
        total_loss = (marg_loss + alpha * rec_loss).mean()
        return total_loss, rec_loss.mean(), marg_loss #.mean()
        
    def margin_loss(self, x, labels):
        batch_size = x.size(0)
        v_c = torch.norm(x, dim=2, keepdim=True)
        
        left = F.relu(0.9 - v_c).view(batch_size, -1) ** 2
        right = F.relu(v_c - 0.1).view(batch_size, -1) ** 2

        loss = labels * left + 0.5 *(1-labels)*right
        loss = loss.sum(dim=1)
        return loss

    def reconstruction_loss(self, data, reconstructions):
        batch_size = reconstructions.size(0)
        reconstructions = reconstructions.view(batch_size, -1)
        data = data.view(batch_size, -1)
        loss = self.reconstruction_criterion(reconstructions, data)
        loss = loss.sum(dim=1)
        return loss
            

# First Convolutional Layer
class ConvLayer(nn.Module):
  def __init__(self, in_channels=1, out_channels=256, kernel_size=9, batchnorm=False):
    super(ConvLayer, self).__init__()
    
    if batchnorm:
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
    else:
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1),
            nn.ReLU())

  def forward(self, x):
    output = self.conv(x)
    return output


class PrimaryCapsules(nn.Module):
  def __init__(self, num_capsules=32, in_channels=256, out_channels=8, kernel_size=9, primary_caps_gridsize=6, batchnorm=False):

    super(PrimaryCapsules, self).__init__()
    self.gridsize = primary_caps_gridsize
    self.num_capsules = num_capsules
    if batchnorm:
        self.capsules = nn.ModuleList([
          nn.Sequential(
          nn.Conv2d(in_channels=in_channels, out_channels=num_capsules, kernel_size=kernel_size, stride=2, padding=0),
          nn.BatchNorm2d(num_capsules))
           for i in range(out_channels)])
    else:
        self.capsules = nn.ModuleList([
          nn.Sequential(
          nn.Conv2d(in_channels=in_channels, out_channels=num_capsules, kernel_size=kernel_size, stride=2, padding=0))
           for i in range(out_channels)])
  
  def forward(self, x):
    output = [caps(x) for caps in self.capsules]
    output = torch.stack(output, dim=1)
    output = output.view(x.size(0), self.num_capsules*(self.gridsize)*(self.gridsize), -1)
    
    return squash(output)     
    


class ClassCapsules(nn.Module):
  
  def __init__(self, num_capsules=10, num_routes = 32*6*6, in_channels=8, out_channels=16, routing_iterations=3, leaky=False):
    super(ClassCapsules, self).__init__()
    
    self.in_channels = in_channels
    self.num_routes = num_routes
    self.num_capsules = num_capsules
    self.routing_iterations = routing_iterations
    
    self.W = nn.Parameter(torch.rand(1, num_routes, num_capsules, out_channels, in_channels))
    self.bias = nn.Parameter(torch.rand(1,1, num_capsules, out_channels))


  # [batch_size, 10, 16, 1]
  def forward(self, x):
    v_j = routing_algorithm(x, self.W, self.bias, self.routing_iterations)
    return v_j.unsqueeze(-1)      


class ReconstructionModule(nn.Module):
  def __init__(self, capsule_size=16, num_capsules=10, imsize=64, img_channel=1, batchnorm=False):
    super(ReconstructionModule, self).__init__()
    
    self.num_capsules = num_capsules
    self.capsule_size = capsule_size
    self.imsize = imsize
    self.img_channel = img_channel
    if batchnorm:
        self.decoder = nn.Sequential(
              nn.Linear(capsule_size*num_capsules, 512),
              nn.BatchNorm1d(512),
              nn.ReLU(),
              nn.Linear(512, 1024),        
              nn.BatchNorm1d(1024),
              nn.ReLU(),
              nn.Linear(1024, imsize*imsize*img_channel),
              nn.Sigmoid()
        )
    else:
        self.decoder = nn.Sequential(
              nn.Linear(capsule_size*num_capsules, 512),
              nn.ReLU(),
              nn.Linear(512, 1024),        
              nn.ReLU(),
              nn.Linear(1024, imsize*imsize*img_channel),
              nn.Sigmoid())
        
  def forward(self, x, target=None):
    batch_size = x.size(0)
    if target is None:
      classes = torch.norm(x, dim=2)
      max_length_indices = classes.max(dim=1)[1].squeeze()
    else:
      max_length_indices = target.max(dim=1)[1]
    
    masked = Variable(x.new_tensor(torch.eye(self.num_capsules)))

    if x.is_cuda:
        masked = masked.cuda()
    
    masked = masked.index_select(dim=0, index=max_length_indices.data)
    decoder_input = (x * masked[:, :, None, None]).view(batch_size, -1)

    reconstructions = self.decoder(decoder_input)
    reconstructions = reconstructions.view(-1, self.img_channel, self.imsize, self.imsize)
    return reconstructions, masked

class ConvReconstructionModule(nn.Module):
  def __init__(self, num_capsules=10, capsule_size=16, imsize=64, img_channel=1, batchnorm=False):
    super(ConvReconstructionModule, self).__init__()
    self.num_capsules = num_capsules
    self.capsule_size = capsule_size
    self.imsize = imsize
    self.img_channel = img_channel
    self.grid_size = 6
    if batchnorm:
      self.FC = nn.Sequential(
        nn.Linear(capsule_size * num_capsules, num_capsules * (self.grid_size)**2 ),
        nn.BatchNorm1d(num_capsules * self.grid_size**2),
        nn.ReLU())
      self.decoder = nn.Sequential(
          nn.ConvTranspose2d(in_channels=self.num_capsules, out_channels=32, kernel_size=9, stride=2),
          nn.BatchNorm2d(32),
          nn.ReLU(),
          nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=9, stride=1),
          nn.BatchNorm2d(64),
          nn.ReLU(),
          nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=2, stride=1),
          nn.Sigmoid())
    else:
        self.FC = nn.Sequential(
            nn.Linear(capsule_size * num_capsules, num_capsules *(self.grid_size**2)),
            nn.ReLU())
        self.decoder = nn.Sequential(
          nn.ConvTranspose2d(in_channels=self.num_capsules, out_channels=32, kernel_size=9, stride=2),
          nn.ReLU(),
          nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=9, stride=1),
          nn.ReLU(),
          nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=2, stride=1),
          nn.Sigmoid())
    
  def forward(self, x, target=None):
    batch_size = x.size(0)
    if target is None:
      classes = torch.norm(x, dim=2)
      max_length_indices = classes.max(dim=1)[1].squeeze()
    else:
      max_length_indices = target.max(dim=1)[1]
    
    masked = x.new_tensor(torch.eye(self.num_capsules))
    masked = masked.index_select(dim=0, index=max_length_indices.data)

    decoder_input = (x * masked[:, :, None, None]).view(batch_size, -1)
    decoder_input = self.FC(decoder_input)
    decoder_input = decoder_input.view(batch_size,self.num_capsules, self.grid_size, self.grid_size)
    reconstructions = self.decoder(decoder_input)
    reconstructions = reconstructions.view(-1, self.img_channel, self.imsize, self.imsize)
    
    return reconstructions, masked


class SmallNorbConvReconstructionModule(nn.Module):
  def __init__(self, num_capsules=10, capsule_size=16, imsize=64, img_channel=1, batchnorm=False):
    super(SmallNorbConvReconstructionModule, self).__init__()
    self.num_capsules = num_capsules
    self.capsule_size = capsule_size
    self.imsize = imsize
    self.img_channel = img_channel
    
    self.grid_size = 4
    
    if batchnorm:
      self.FC = nn.Sequential(
            nn.Linear(capsule_size * num_capsules, num_capsules *self.grid_size*self.grid_size),
            nn.BatchNorm1d(num_capsules * self.grid_size**2),
            nn.ReLU())
      self.decoder = nn.Sequential(
          nn.ConvTranspose2d(in_channels=num_capsules, out_channels=32, kernel_size=9, stride=2),
          nn.BatchNorm2d(32),            
          nn.ReLU(),
          nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=9, stride=1),
          nn.BatchNorm2d(64),
          nn.ReLU(),
          nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=9, stride=1),
          nn.BatchNorm2d(128),
          nn.ReLU(),
          nn.ConvTranspose2d(in_channels=128, out_channels=img_channel, kernel_size=2, stride=1),
          nn.Sigmoid())
    else:
        self.FC = nn.Sequential(
            nn.Linear(capsule_size * num_capsules, num_capsules *(self.grid_size**2) ),
            nn.ReLU())
        self.decoder = nn.Sequential(
          nn.ConvTranspose2d(in_channels=num_capsules, out_channels=32, kernel_size=9, stride=2),
          nn.ReLU(),
          nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=9, stride=1),
          nn.ReLU(),
          nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=9, stride=1),
          nn.ReLU(),
          nn.ConvTranspose2d(in_channels=128, out_channels=img_channel, kernel_size=2, stride=1),
          nn.Sigmoid())
    
  def forward(self, x, target=None):
    batch_size = x.size(0)
    if target is None:
      classes = torch.norm(x, dim=2)
      max_length_indices = classes.max(dim=1)[1].squeeze()
    else:
      max_length_indices = target.max(dim=1)[1]
    masked = Variable(x.new_tensor(torch.eye(self.num_capsules)))
    masked = masked.index_select(dim=0, index=max_length_indices.data)

    decoder_input = (x * masked[:, :, None, None]).view(batch_size, -1)
    decoder_input = self.FC(decoder_input)
    decoder_input = decoder_input.view(batch_size,self.num_capsules, self.grid_size, self.grid_size)
    reconstructions = self.decoder(decoder_input)
    reconstructions = reconstructions.view(-1, self.img_channel, self.imsize, self.imsize)
    
    return reconstructions, masked


# The squash function specified in Dynamic Routing Between Capsules
# x: input tensor 
def squash(x, dim=-1):
  norm_squared = (x ** 2).sum(dim, keepdim=True)
  part1 = norm_squared / (1 +  norm_squared)
  part2 = x / torch.sqrt(norm_squared+ 1e-16)

  output = part1 * part2 
  return output
   

def routing_algorithm(x, weight, bias, routing_iterations):
    """
    x: [batch_size, num_capsules_in, capsule_dim]
    weight: [1, num_capsules_in, num_capsules_out, out_channels, in_channels]
    bias: [1, 1, num_capsules_out, out_channels]
    """
    num_capsules_in = x.shape[1]
    num_capsules_out = weight.shape[2]
    batch_size = x.size(0)
    
    x = x.unsqueeze(2).unsqueeze(4) # x=> (128, 2048, 8) -> (128, 2048, 1, 8, 1)

    #[batch_size, 32*6*6, 10, 16]
    u_hat = torch.matmul(weight, x).squeeze() #weight => (128, 2048, 5, 16, 8)

    b_ij = Variable(x.new(batch_size, num_capsules_in, num_capsules_out, 1).zero_())

    for it in range(routing_iterations):
      c_ij = F.softmax(b_ij, dim=2)    #  c_ij, b_ij => (128, 2048, 5, 1)

      # [batch_size, 1, num_classes, capsule_size]
      s_j = (c_ij * u_hat).sum(dim=1, keepdim=True) + bias #u_hat => (128, 2048, 5, 16)
      # [batch_size, 1, num_capsules, out_channels]         s_j   => (128,    1, 5, 16)   
      v_j = squash(s_j, dim=-1)                            #v_j   => (128,    1, 5, 16)
      
      if it < routing_iterations - 1: 
        # [batch-size, 32*6*6, 10, 1]
        delta = (u_hat * v_j).sum(dim=-1, keepdim=True)
        b_ij = b_ij + delta
    
    return v_j.squeeze(1)
