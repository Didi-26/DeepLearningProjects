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

    
class LeNet5Like(nn.Module):
    
    """ Modified LeNet5 (adapted to have 14*14 inputs & ouptut size 2) """
    
    def __init__(self, dropout=False):
        super(LeNet5Like, self).__init__()
        self.features = nn.Sequential(
              # (N, 2, 14, 14) -> (N, 6, 10, 10)
              nn.Conv2d(2, 6, kernel_size=5),
              nn.ReLU(),
              # (N, 6, 10, 10) -> (N, 6, 5, 5)
              nn.MaxPool2d(kernel_size=2, stride=2),
              # (N, 6, 5, 5) -> (N, 16, 1, 1)
              nn.Conv2d(6, 16, kernel_size=5),
              nn.ReLU()
          )  
        if dropout :
            self.classifier = nn.Sequential(
                  # (N, 16) -> (N, 8)
                  nn.Linear(16, 8),
                  nn.ReLU(),
                  nn.Dropout(0.1),
                  # (N, 8) -> (N, 4)
                  nn.Linear(8, 4),
                  nn.ReLU(),
                  nn.Dropout(0.1),
                  # (N, 4) -> (N, 2)
                  nn.Linear(4, 2)
              )
        else :
            self.classifier = nn.Sequential(
                  # (N, 16) -> (N, 8)
                  nn.Linear(16, 8),
                  nn.ReLU(),
                  # (N, 8) -> (N, 4)
                  nn.Linear(8, 4),
                  nn.ReLU(),
                  # (N, 4) -> (N, 2)
                  nn.Linear(4, 2),
                  nn.ReLU()
              )
        
    def forward(self, x):
        # Feature block
        x = self.features(x)
        # Flatten to vector before linear layers
        x = x.view(x.size(0), -1)
        # Classifier block
        x = self.classifier(x)
        return x

    
class LeNet3(nn.Module):
    
    """ Inspired from LeNet5 but with smaller 3*3 convolution """
    
    def __init__(self):
        super(LeNet3, self).__init__()
        self.features = nn.Sequential(
              # (N, 2, 14, 14) -> (N, 6, 12, 12)
              nn.Conv2d(2, 8, kernel_size=3),
              nn.ReLU(),
              # (N, 8, 12, 12) -> (N, 8, 6, 6)
              nn.MaxPool2d(kernel_size=2, stride=2),
              # (N, 8, 6, 6) -> (N, 16, 4, 4)
              nn.Conv2d(8, 16, kernel_size=3),
              nn.ReLU(),
              # (N, 16, 4, 4) -> (N, 16, 2, 2)
              nn.MaxPool2d(kernel_size=2, stride=2),
              nn.ReLU(),
              # (N, 16, 2, 2) -> (N, 32, 1, 1)
              nn.Conv2d(16, 32, kernel_size=2),
              nn.ReLU()
          )
        self.classifier = nn.Sequential(
            # (N, 32) -> (N, 16)
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            # (N, 16) -> (N, 8)
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(0.1),
            # (N, 8) -> (N, 2)
            nn.Linear(8, 2)
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
    def __init__(self, nb_channels, kernel_size):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(nb_channels, nb_channels, kernel_size,padding = (kernel_size-1)//2)
        self.bn1 = nn.BatchNorm2d(nb_channels)
        self.conv2 = nn.Conv2d(nb_channels, nb_channels, kernel_size,padding = (kernel_size-1)//2)
        self.bn2 = nn.BatchNorm2d(nb_channels)
    def forward(self, x):
        y = self.bn1(self.conv1(x))
        y = F.relu(y)
        y = self.bn2(self.conv2(y))
        y += x
        y = F.relu(y)
        return y
class ResNet(nn.Module):
    def __init__(self, nb_channels, kernel_size, nb_blocks):
        super(ResNet, self).__init__()
        self.conv0 = nn.Conv2d(1, nb_channels, kernel_size = 1)
        self.resblocks = nn.Sequential(
            # A bit of fancy Python
            *(ResBlock(nb_channels, kernel_size) for _ in range(nb_blocks)))
        self.avg = nn.AvgPool2d(kernel_size = 28)
        self.fc = nn.Linear(nb_channels, 10)
    def forward(self, x):
        x = F.relu(self.conv0(x))
        x = self.resblocks(x)
        x = F.relu(self.avg(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    

class VGGNetLike(nn.Module):
    
    """ Inspired from VGGNet, adapted to have 14*14 inputs & ouptut size 2) """
    
    def __init__(self, dropout):
        super(VGGNetLike, self).__init__()
        layers = []
        # We start from 14*14 pictures padded to 16*16 and divide the 
        # dimension by two while multiplying the channel count by two 
        # at each block
        layers.append(nn.ConstantPad2d(1, 0))
        for l in range(4):
            # We go to 64 channels and multiply by 2 the number of
            # channel at each block (like in original VGGNet) 
            channels_count = 2**(6+l)
            prev_channels_count = 2 if (l == 0) else 2**(6+(l-1))
            print((prev_channels_count, channels_count))
            layers.append(nn.Conv2d(prev_channels_count, channels_count, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            #layers.append(nn.Conv2d(channels_count, channels_count, kernel_size=3, padding=1))
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))   
        self.features = nn.Sequential(*layers)
        # Classifier with fully connected layers
        if dropout :
            self.classifier = nn.Sequential(
                  # (N, 512) -> (N, 128)
                  nn.Linear(512, 128),
                  nn.ReLU(),
                  nn.Dropout(0.5),
                  # (N, 128) -> (N, 64)
                  nn.Linear(128, 64),
                  nn.ReLU(),
                  nn.Dropout(0.5),
                  # (N, 64) -> (N, 2)
                  nn.Linear(64, 2)
              )
        else :
            self.classifier = nn.Sequential(
                  # (N, 512) -> (N, 128)
                  nn.Linear(512, 128),
                  nn.ReLU(),
                  # (N, 128) -> (N, 64)
                  nn.Linear(128, 64),
                  nn.ReLU(),
                  # (N, 64) -> (N, 2)
                  nn.Linear(64, 2)
              )
        
    def forward(self, x):
        # Feature block
        x = self.features(x)
        # Flatten to vector before linear layers
        x = x.view(x.size(0), -1)
        # Classifier block
        x = self.classifier(x)
        return x