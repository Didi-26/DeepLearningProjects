# code for the different functions that we use
import torch
from torch import Tensor
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import dlc_practical_prologue as prologue
from torch.optim.lr_scheduler import MultiStepLR

import matplotlib.pyplot as plt

def normalization_(train_input,test_input):
    """The input data is normalized. This function modifies the input data"""
    mu = train_input.mean(dim=0)
    std = train_input.std(dim=0)
    train_input.sub_(mu).div_(std+1)
    test_input.sub_(mu).div_(std+1)

def compute_errors(output, target):
    """ Computes error percentage given output and target"""
    errors_amount = (output.argmax(dim=1) != target.argmax(dim=1)).sum().item()
    return (errors_amount / output.shape[0]) * 100
def compute_errors2(output,target):
    errors_amount = (output.argmax(dim=1) != target).sum().item()
    return (errors_amount / output.shape[0]) * 100

def train(model, train_input, train_target, test_input, test_target,test_target_final,criteria=nn.MSELoss(),lr=1e-3, batch_size=100, epochs=200) :
    """Train the model given the different parameters and datasets"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    criterion = criteria

    train_errors = []
    train_errors_final = []
    test_errors = []
    test_errors_final = []

    scheduler = MultiStepLR(optimizer, milestones=[10,200], gamma=0.1)

    for e in range(epochs):
        #scheduler.step()
        #("Epoch: {}".format(e))
        print('.', end='')
        for b in range(0, train_input.size(0), batch_size):
            output = model(train_input.narrow(0, b, batch_size))
            train_target_batch = train_target.narrow(0, b, batch_size)
            loss = criterion(output, train_target_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            # Compute train error
            output_train = model(train_input)
            train_errors.append(compute_errors2(output_train, train_target))
            # Compute the train smaller/bigger prediction
            train_errors_final.append(compute_errors_final(output_train,train_target_final))
            # Compute test error
            output_test = model(test_input)
            test_errors.append(compute_errors2(output_test, test_target))
            # Compute the test smaller/bigger prediction
            test_errors_final.append(compute_errors_final(output_test,test_target_final))

    return train_errors, test_errors, train_errors_final,test_errors_final
