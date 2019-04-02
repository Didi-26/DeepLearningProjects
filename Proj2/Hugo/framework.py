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
    else:
        raise ValueError("nonlinearity must be 'none', 'relu' or 'tanh'")
    std = gain * math.sqrt(2.0 / (tensor.size(1)+tensor.size(0)))
    return tensor.normal_(0, std)


#
# - Modules -
#

# Convention : any tensor (whereas it's data (forward) of gradients (backward))
# always has a dimension of N*D where N are the number of samples of the batch
# and D the dimentions of the sample (at that stage of the network).


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
        # Average over over batch samples of gradients
        self.w_grad = torch.empty(out_feature_nb, in_feature_nb)
        self.bias_grad = torch.empty(out_feature_nb)
        # Parameter (weights and bias) values
        self.w = torch.empty(out_feature_nb, in_feature_nb)
        self.bias = torch.empty(out_feature_nb)
        # Initialize parameters
        xavier_init(self.w, nonlinearity_init)
        self.bias.zero_()

    def forward(self, input):
        self.last_input = input.clone()
        return input @ self.w.t() + self.bias

    def backward(self, gradwrtoutput):
        # Average over batch samples of loss gradient wrt bias
        self.bias_grad.data = gradwrtoutput.mean(dim=0)
        # We want to compute a w grad matrix for each sample N of the batch
        N = self.last_input.shape[0]
        last_input_view = self.last_input.view(N, 1, self.in_feature_nb)
        gradwrtoutput_view = gradwrtoutput.view(N, self.out_feature_nb, 1)
        # Average over batch samples of loss gradient wrt w
        self.w_grad.data = (gradwrtoutput_view @ last_input_view).mean(dim=0)
        return gradwrtoutput @ self.w  # loss gradient wrt input

    def param(self):
        return [(self.w, self.w_grad), (self.bias, self.bias_grad)]


class ReLU(Module):

    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, input):
        self.last_input = input.clone()
        # relu(x) is 0 if x < 0, x otherwise
        return ((input.sign()+1) / 2) * input

    def backward(self, gradwrtoutput):
        # relu(x) derivative is 0 if x < 0, 1 otherwise
        return ((self.last_input.sign()+1) / 2) * gradwrtoutput

    def param(self):
        return []


# TODO tanh


class Sequential(Module):

    def __init__(self, modules_array):
        super(Sequential, self).__init__()
        self.modules_array = modules_array

    def forward(self, input):
        for module in self.modules_array:
            input = module.forward(input)
        return input

    def backward(self, gradwrtoutput):
        for module in reversed(self.modules_array):
            gradwrtoutput = module.backward(gradwrtoutput)
        return gradwrtoutput

    def param(self):
        parameter_array = []
        for module in self.modules_array:
            parameter_array.append(module.param())
        return parameter_array


class LossMSE(Module):

    def __init__(self, targets):
        super(LossMSE, self).__init__()
        self.targets = targets
        self.N = targets.shape[0]  # batch size

    def forward(self, input):
        self.last_input = input.clone()
        return ((self.targets - input)**2).sum(dim=1).mean()

    def backward(self, gradwrtoutput=1):
        # loss (MSE) derivative wrt input
        return gradwrtoutput * (-2.0/self.N) * (self.targets - self.last_input)

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


class SGD(Optimizer):

    def __init__(self, parameters, lr):
        super(SGD, self).__init__()
        self.parameters = parameters
        self.lr = lr  # learning rate

    def zero_grad(self):
        # Set gradients of all parameters to zero
        for p in self.parameters:
            for pair in p:
                pair[1].zero_()

    def step(self):
        # Update parameters according to the average of the gradients of all
        # samples
        for p in self.parameters:
            for pair in p:
                pair[0].data -= self.lr*pair[1]
