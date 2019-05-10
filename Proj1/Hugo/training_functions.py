import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import dlc_practical_prologue as prologue
import numpy as np

import plots


class TrainDataset(Dataset):
    """ 
    PyTorch Dataset for holding MNIST train pairs (PairSetup) or single pictures (AuxiliarySetup). 
    Arguments: 
        - train_input: a torch tensor of size [N, 2, 14, 14] for PairSetup or [2*N, 14, 14] 
        for AuxiliarySetup
        - train_target: hot-encoded target torch tensor of size [N , 2] for PairSetup or [2*N , 10]
        for AuxiliarySetup
        - augment_data: boolean, if True, data will be augmented by inversing pairs. Can be true only
        for PairSetup
    """
    def __init__(self, train_input, train_target, augment_data=False):
        if augment_data :
            # Create the inversed pairs (data augmentation)
            train_input_rev = train_input[:,[1,0],:,:]
            train_target_rev = train_target[:,[1,0]]
            self.train_input = torch.cat((train_input,train_input_rev))
            self.train_target = torch.cat((train_target,train_target_rev))
        else :
            self.train_input = train_input
            self.train_target = train_target
            
    def __len__(self):
        return len(self.train_input)

    def __getitem__(self, idx):
        return {'input': self.train_input[idx], 'target': self.train_target[idx]}


def hot_decode(data):
    """
    Hot decoding i.e return argmax of each row.
    Arguments:
        - data: hot-encoded torch tensor of size [N,D]
    Returns:
        - hot-decoded torch tensor of size N
    """
    return torch.argmax(data, dim=1).long()


def compute_errors(output, target):
    """ 
    Computes error percentage given output and target
    Arguments:
        - output: torch tensor of [N,2] for PairSetup or [2*N, 10] for AuxiliarySetup 
        of predicted scores for each class
        - target: hot-encoded target torch tensor of size [N,2] of [N,2] for PairSetup 
        or [2*N, 10] for AuxiliarySetup
    Returns:
        - error %
    """
    errors_amount = (output.argmax(dim=1) != target.argmax(dim=1)).sum().item()
    return (errors_amount / output.shape[0]) * 100


def train(model, setup, train_input, train_target, test_input, test_target,
          use_crossentropy = False, lr=1e-3, epochs = 200, verbose=False) :
    """ 
    Trains the given model using the given train and test dataset. Returns the
    train & test error % history.
    Arguments:
        - model: torch model to train
        - setup: str, either 'PairSetup' or 'AuxiliarySetup'. In the first one we consider pairs and
        0 or 1 targets. In the second we take individual pictures as input nd train to recognize the 
        digits (i.e targets are 0,1...,8 or 9)
        - train_input: torch tensor of train input data
        - train_target: hot-encoded train target class
        - test_input: torch tensor of test input data
        - test_target: hot-encoded test target class
        - use_crossentropy: boolean, if True, crossentropy loss will be used (train 
        target data will be dencoded in order to use this loss). If False MSE loss 
        will be used.
        - lr: learning rate
        - epochs: number of epochs to train with
        - verbose: if True, a dot '.' will be printed at each new epoch
    Returns:
        - (train_errors, test_errors), the train and test error % histories
    """ 
    N = train_input.shape[0]
    
    if setup == 'PairSetup':
            # Hot encoding (train target will be later hot decoded if the loss is cross-entropy)
            train_target = nn.functional.one_hot(train_target, 2).float()
            test_target = nn.functional.one_hot(test_target, 2).float()
            
    if setup == 'AuxiliarySetup':
            # Split pairs (flatten) into individual pictures
            train_input = train_input.reshape((2*N, 14, 14))
            test_input = test_input.reshape((2*N, 14, 14))
            # Flatten targets
            train_aux_target = train_aux_target.reshape(2*N)
            test_aux_target = test_aux_target.reshape(2*N)
            # Hot encoding (train targets will be later hot decoded if the loss is cross-entropy)
            train_aux_target = nn.functional.one_hot(train_aux_target, 10).float()
            test_aux_target = nn.functional.one_hot(test_aux_target, 10).float()
    
    batch_size = 100
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) 
    criterion = nn.CrossEntropyLoss() if use_crossentropy else nn.MSELoss()
    
    trainDataset = TrainDataset(train_input, train_target)
    dataloader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True, 
                            num_workers=0)
    train_errors = []
    test_errors = []
    
    for e in range(epochs):
        if verbose :
            print('.', end='')
        for i_batch, batch in enumerate(dataloader):
            inputSamples = batch['input']
            # If we use crossentropy, then we don't want hot-encoding of train target class but 
            # directly their class.
            target = hot_decode(batch['target']) if use_crossentropy else batch['target']
            # Forward pass
            output = model(inputSamples)
            # Compute loss
            loss = criterion(output, target)
            # Backprop & update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        with torch.no_grad():
            # Compute train error
            output_train = model(train_input)
            train_errors.append(compute_errors(output_train, train_target))
            # Compute test error
            output_test = model(test_input)
            test_errors.append(compute_errors(output_test, test_target))
                
    return train_errors, test_errors


def rounds_train(model, setup, rounds=10, augment_data=False, use_crossentropy=True, 
                          lr=1e-3, epochs=200, verbose=False, plot_title = None,
                          plot_file_path = None):
    """
    Trains the model multiple times with randomized fresh new data. For each training, we keep
    only the best test error (early stopping) with its corresponding train error. Then we compute
    and return the average and standard deviation of those best test errors and their corresponding 
    train errors.
    Arguments:
        - model: torch model to train
        - setup: str, either 'PairSetup' or 'AuxiliarySetup'. In the first one we consider pairs and
        0 or 1 targets. In the second we take individual pictures as input nd train to recognize the 
        digits (i.e targets are 0,1...,8 or 9)
        - rounds: number of experiment (new training) to perform
        - augment_data : boolean, if True, data will be augmented by inversing pairs
        - use_crossentropy: boolean, if True, crossentropy loss will be used. If False MSE loss will 
        be used.
        - lr: learning rate
        - epochs: number of epochs to train with at each round
        - verbose: if True, the current round will be printed at each round
        - plot_title : str, if plot_title and plot_file_path are not None, then a figure with 
        the given titlt will be saved at the given path
        - plot_file_path : str, if plot_title and plot_file_path are not None, then a figure with 
        the given titlt will be saved at the given path
    Returns:
        - (min_test_mean, min_test_std, min_tran_mean, min_train_std), the average and standard deviation
        of the min-train (and corresponding test) errors %.
    """
    min_test_errors = []
    corresponding_train_errors = []
    train_errors_histories = []
    test_errors_histories = []
    
    # Number of samples to generate each round (=1000 as per the project instructions)
    N=1000  
    
    for i in range(0, rounds):
        if verbose :
            print('round n°{}'.format(i+1))
        # Load new data (data is randomized at each round as required in the project instructions)
        train_input, train_target, train_aux_target , test_input, test_target, test_aux_target = prologue.generate_pair_sets(N)
        # Train
        train_errors, test_errors = train(model,
                                          setup,
                                          train_input, train_target, 
                                          test_input, test_target, 
                                          use_crossentropy=use_crossentropy, lr=lr, epochs=epochs)
        # Store error histories
        train_errors_histories.append(train_errors)
        test_errors_histories.append(test_errors)
        # Use use early stopping, we take the train & test error where the test error was the smallest
        min_test_errors.append(min(test_errors))
        corresponding_train_errors.append(train_errors[test_errors.index(min(test_errors))])
    
    # Plot figure
    if (plot_title != None) and (plot_file_path != None):
        plots.plot_errors(train_errors_histories, test_errors_histories, plot_title, plot_file_path)
    
    # Compute and return mean/std of min test error and its corresponding train error    
    return (np.mean(min_test_errors), np.std(min_test_errors), 
            np.mean(corresponding_train_errors), np.std(corresponding_train_errors))