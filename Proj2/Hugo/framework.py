import torch
from torch import empty
import math
torch.set_grad_enabled(False)


#
# - Util functions -
#


def xavier_init(tensor, nonlinearity):
    if nonlinearity == 'none':
        gain = 1
    elif nonlinearity == 'tanh':
        gain = 5.0 / 3
    elif nonlinearity == 'relu':
        gain = math.sqrt(2.0)
    else :
        raise ValueError("nonlinearity must be 'none', 'relu' or 'tanh'")  
    std = gain * math.sqrt(2.0 / (tensor.size(1)+tensor.size(0)))
    return tensor.normal_(0, std)


#
# - Modules -
#

# Convention : any tensor (whereas it is data (forward) of gradients (backward))
# always has a dimension of N*D where N are the number of samples of the batch 
# and D the dimentions of the sample (at that stage of the network).
# ( the course slides are using the Jacobian notation i.e D*N )


class Module(object):
    
    def forward(self, *input):
        raise NotImplementedError
        
    def backward(self, *gradwrtoutput):
        raise NotImplementedError
        
    def param(self):
        return []


class Linear(Module):
    
    def __init__(self, in_feature_nb, out_feature_nb, nonlinearity_init):
        super(Linear, self).__init__()
        self.in_feature_nb = in_feature_nb
        self.out_feature_nb = out_feature_nb
        # Parameter (weights and bias) values
        self.w = torch.empty(out_feature_nb, in_feature_nb)
        self.bias = torch.empty(out_feature_nb)
        # Initialize parameters
        xavier_init(self.w, nonlinearity_init)
        self.bias.zero_()
        
    def forward(self, input):
        self.last_input = input
        return input @ self.w.t() + self.bias
        
    def backward(self, gradwrtoutput):
        self.bias_grad = gradwrtoutput # loss gradient wrt bias
        self.w_grad =  gradwrtoutput.t() @ self.last_input # loss gradient wrt w
        return (self.w.t() @ gradwrtoutput.t()) # loss gradient wrt input
        
    def param(self):
        return [(self.w, self.w_grad), (self.bias, self.bias_grad)]
    
    
class ReLu(Module):
    
    def __init__(self):
        super(ReLu, self).__init__()
        
    def forward(self, input):
        # relu(x) is 0 if x < 0, x otherwise
        return ( (a.sign()+1) / 2 ) * a
    
    def backward(self, gradwrtoutput):
        # relu(x) derivative is 0 if x < 0, 1 otherwise
        return (a.sign()+1) / 2
    
    def param(self):
        return []
    
    
#
# - Optimizer -
#


class Optimizer(object):
    
    def zero_grad(self):
        raise NotImplementedError
        
    def step():
        raise NotImplementedError
    