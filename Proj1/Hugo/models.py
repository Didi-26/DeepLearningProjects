import torch
from torch import nn


class SimpleLinear(nn.Module):
    """
    A simple Linear Model.
    Arguments:
        - in_dim: dimension of input
        - out_dim: dimension of output
    """
    def __init__(self, in_dim, out_dim):
        super(SimpleLinear, self).__init__()
        # (N, in_dim) -> (N, out_dim)
        self.layer = nn.Linear(in_dim, out_dim)
        
    def forward(self, x):
        # Flatten to vector before linear layer
        x = x.view(x.size(0), -1)
        # Linear
        x = self.layer(x)
        return x


class MLP(nn.Module):
    """ A multilayer perceptron (i.e fully connected + ReLu layers) of L hidden 
    layers of with h hidden neurons per layer 
    Arguments:
        - in_dim: dimension of input
        - out_dim: dimension of output
    """
    def __init__(self, L, h, in_dim, out_dim):
        super(MLP, self).__init__()
        # (N, in_dim) -> (N, h)
        self.in_layer = nn.Sequential(nn.Linear(in_dim, h),
                                      nn.ReLU())
        # (N, h) -> (N, h) -> ... -> (N, h)
        hidden_layers = []
        for l in range(L) :
            hidden_layers.append(nn.Dropout(0.02))
            hidden_layers.append(nn.Linear(h,h))
            hidden_layers.append(nn.ReLU())
        self.hidden_layers = nn.Sequential(*hidden_layers)
        # (N, h) -> (N, out_dim)
        self.out_layer = nn.Linear(h, out_dim)
        
    def forward(self, x):
        # Flatten to vector before linear layers
        x = x.view(x.size(0), -1)
        # MLP net
        x = self.in_layer(x)
        x = self.hidden_layers(x)
        x = self.out_layer(x)
        return x

    
class LeNetLike5(nn.Module): 
    """ Inspired from LeNet5. Input tensor of size N*in_depth*14*14, and output
    tensor of size N*out_dim. Convolution kernels are 5*5.
    Arguments:
        - in_depth : number of channels of input picture (2 for PairSetup and 1 for
    AuxiliarySetup)
        - out_dim : dimension of the output (2 for PairSetup and 10 for 
    AuxiliarySetup)
    """
    def __init__(self, in_depth, out_dim):
        super(LeNetLike5, self).__init__()
        self.features = nn.Sequential(
              # (N, in_depth, 14, 14) -> (N, 6, 10, 10)
              nn.Conv2d(in_depth, 6, kernel_size=5),
              nn.ReLU(),
              # (N, 6, 10, 10) -> (N, 6, 5, 5)
              nn.MaxPool2d(kernel_size=2, stride=2),
              # (N, 6, 5, 5) -> (N, 16, 1, 1)
              nn.Conv2d(6, 16, kernel_size=5),
              nn.ReLU()
        )  
        self.classifier = nn.Sequential(
             # (N, 16) -> (N, 14)
             nn.Linear(16, 14),
             nn.ReLU(),
             # (N, 14) -> (N, 10)
             nn.Linear(14, 10),
             nn.ReLU(),
             # (N, 10) -> (N, out_dim)
             nn.Linear(10, out_dim)
         )
        
    def forward(self, x):
        # Feature block
        x = self.features(x)
        # Flatten to vector before linear layers
        x = x.view(x.size(0), -1)
        # Classifier block
        x = self.classifier(x)
        return x

    
class LeNetLike3(nn.Module): 
    """ Inspired from LeNet5. Input tensor of size N*in_depth*14*14, and output
    tensor of size N*out_dim. Convolution kernels are 3*3.
    Arguments:
        - in_depth : number of channels of input picture (2 for PairSetup and 1 for
    AuxiliarySetup)
        - out_dim : dimension of the output (2 for PairSetup and 10 for 
    AuxiliarySetup)
    """
    def __init__(self, in_depth, out_dim):
        super(LeNetLike3, self).__init__()
        self.features = nn.Sequential(
              # (N, in_depth, 14, 14) -> (N, 6, 12, 12)
              nn.Conv2d(in_depth, 6, kernel_size=3),
              nn.ReLU(),
              # (N, 6, 12, 12) -> (N, 6, 6, 6)
              nn.MaxPool2d(kernel_size=2, stride=2),
              # (N, 6, 6, 6) -> (N, 16, 4, 4)
              nn.Conv2d(6, 16, kernel_size=3),
              nn.ReLU(),
              # (N, 16, 4, 4) -> (N, 16, 2, 2)
              nn.MaxPool2d(kernel_size=2, stride=2)
        )  
        self.classifier = nn.Sequential(
             # (N, 64) -> (N, 32)
             nn.Linear(64, 32),
             nn.ReLU(),
             # (N, 32) -> (N, 16)
             nn.Linear(32, 16),
             nn.ReLU(),
             # (N, 16) -> (N, out_dim)
             nn.Linear(16, out_dim)
         )
        
    def forward(self, x):
        # Feature block
        x = self.features(x)
        # Flatten to vector before linear layers
        x = x.view(x.size(0), -1)
        # Classifier block
        x = self.classifier(x)
        return x

    
class VGGNetLike(nn.Module): 
    """ Inspired from VGGNet but less heavy : instead of going to from 
    64 to 512 channels like in the original VGGNet19, we go from 16 to 128 channels.
    Also we only do one convolution per block while VGGNet19 does two.
    Input tensor of size N*in_depth*14*14, and output tensor of size N*out_dim. 
    Convolution kernels are 3*3.
    Arguments:
        - in_depth : number of channels of input picture (2 for PairSetup and 1 for
    AuxiliarySetup)
        - out_dim : dimension of the output (2 for PairSetup and 10 for 
    AuxiliarySetup)
    """
    def __init__(self, in_depth, out_dim):
        super(VGGNetLike, self).__init__()
        layers = []
        # (N, in_depth, 14, 14) -> (N, in_depth, 16, 16)
        layers.append(nn.ConstantPad2d(1, 0))
        # (N, in_depth, 16, 16) -> (N, 16, 8, 8) -> (N, 32, 4, 4) -> (N, 64, 2, 2) -> (N, 128, 1, 1)
        for l in range(4): 
            # First block goes to 16 channels (instead of 64 in original VGG) 
            # and halfs the width/height
            # The 3 other block double the channels and halfs the width/height
            channels_count = 2**(4+l)
            prev_channels_count = in_depth if (l == 0) else 2**(4+(l-1))
            #print((prev_channels_count, channels_count))
            layers.append(nn.Conv2d(prev_channels_count, 
                                    channels_count, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))   
        self.features = nn.Sequential(*layers)
        # Classifier with fully connected layers
        self.classifier = nn.Sequential(            
            # (N, 128) -> (N, 64)
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            # (N, 64) -> (N, 32)
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            # (N, 32) -> (N, out_dim)
            nn.Linear(32, out_dim)
        )
        
    def forward(self, x):
        # Feature block
        x = self.features(x)
        # Flatten to vector before linear layers
        x = x.view(x.size(0), -1)
        # Classifier block
        x = self.classifier(x)
        return x


class ResBlock(nn.Module):
    """
    Residual Block used by Residual Net. From corresponds to the 
    'No ReLU, BN before add' from http://torch.ch/blog/2016/02/04/resnets.html. 
    Convolution kernel are 3*3.
    Arguments :
        - in_depth : number of channels of input
    """
    def __init__(self, depth):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(nn.Conv2d(depth, depth, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(depth),
                                   nn.ReLU(),
                                   nn.Conv2d(depth, depth, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(depth))
        
    def forward(self, x):
        y = self.block(x)
        return y + x    


class ResNet(nn.Module):
    def __init__(self, in_depth, out_dim, net_depth):
        super(ResNet, self).__init__()
        # (N, 1 or 2, 14, 14) -> (N, net_depth, 14, 14)
        self.conv0 = nn.Conv2d(in_depth, net_depth, kernel_size = 1)
        # Res Blocks
        # (N, net_depth, 14, 14) ->  (N, net_depth*2, 7, 7)
        self.resblocks = nn.Sequential(ResBlock(net_depth),
                                       nn.Conv2d(net_depth, net_depth*2, kernel_size=3, padding=1),
                                       nn.ReLU(),
                                       nn.MaxPool2d(kernel_size=2, stride=2),
                                       ResBlock(net_depth*2))
        self.classifier = nn.Sequential(
            # (N, net_depth*2*7*7) -> (N, 128)
            nn.Linear(net_depth*2*7*7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            # (N, 128) -> (N, 64)
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            # (N, 64) -> (N, out_dim)
            nn.Linear(64, out_dim))
                                                
    def forward(self, x):
        # 1*1 convolution to go to net_depth channels
        x = self.conv0(x)
        # Residual blocks
        x = self.resblocks(x)
        # Flatten to vector before linear classifier
        x = x.view(x.size(0), -1)
        # Classifier block
        x = self.classifier(x)
        return x