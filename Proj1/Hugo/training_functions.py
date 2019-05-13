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
        - train_input: a torch tensor of size N*2*14*14 for PairSetup or 2N*1*14*14 
        for AuxiliarySetup
        - train_target: target tensor
        - data_augment: boolean, if True, data will be augmented by inversing pairs. Can be true only
        for PairSetup
    """
    def __init__(self, train_input, train_target, data_augment=False):
        if data_augment :
            # Create the inversed pairs (data augmentation)
            train_input_rev = train_input[:,[1,0],:,:]
            train_target_rev = -(train_target-1)
            self.train_input = torch.cat((train_input,train_input_rev))
            self.train_target = torch.cat((train_target,train_target_rev))
        else :
            self.train_input = train_input
            self.train_target = train_target
            
    def __len__(self):
        return len(self.train_input)

    def __getitem__(self, idx):
        return {'input': self.train_input[idx], 'target': self.train_target[idx]}

    
def compute_error(output, target):
    """ 
    Computes error percentage given prediction scores and hot-encoded target tensor.
    Arguments:
        - output: prediction scores tensor
        - target: hot-encoded target class tensor
    Returns:
        - error %
    """
    errors_amount = (output.argmax(dim=1) != target.argmax(dim=1)).sum().item()
    return (errors_amount / output.shape[0]) * 100


def compute_error_class(output, target):
    """ 
    Computes error percentage given predicted non hot-encoded class and target non 
    hot-encoded class.
    Arguments:
        - output: predicted class tensor
        - target: target class tensor
    Returns:
        - error %
    """
    errors_amount = (output != target).sum().item()
    return (errors_amount / output.shape[0]) * 100


def train(model, setup, 
          train_input_original, train_pair_target, train_aux_target,
          test_input_original, test_pair_target, test_aux_target,
          use_crossentropy = False, lr=1e-3, epochs = 200, verbose=False, data_augment=False) :
    """ 
    Trains the given model using the given train and test dataset. Returns the
    train & test error % history.
    Arguments:
        - model: torch model to train
        - setup: str, either 'PairSetup' or 'AuxiliarySetup'. In the first one we consider pairs and
        0 or 1 targets. In the second we take individual pictures as input nd train to recognize the 
        digits (i.e targets are 0,1...,8 or 9)
        - train_input_original: tensor of train input data
        - train_pair_target: 0 or 1  train target class
        - train_aux_target: 0, 1, ..., 8 or 9 auxiliary train target class
        - test_input_original: tensor of test input data
        - test_pair_target: 0 or 1  test target class
        - test_aux_target: 0, 1, ..., 8 or 9 auxiliary test target class
        - use_crossentropy: boolean, if True, crossentropy loss will be used (train 
        target data will be dencoded in order to use this loss). If False MSE loss 
        will be used.
        - lr: learning rate
        - epochs: number of epochs to train with
        - verbose: if True, a dot '.' will be printed at each new epoch
        - data_augment: bool, if True pairs will be augmented by inversing them. Only works for PairSetup and cross-entropy loss
    Returns:
        - (train_errors, test_errors), the train and test error % histories. Whereas the setup
        is 'PairSetup' or 'AuxiliarySetup', in both case the error is with respect to the final
        class (0 or 1) but not the auxiliary class (0,1,...,8,9) so that the errors are easily
        comparable.
    """ 
    N = train_input_original.shape[0]
    
    if setup == 'PairSetup':
        train_input = train_input_original
        test_input = test_input_original
        train_target = train_pair_target
        test_target = test_pair_target
        # Hot encoding
        train_target_hot = nn.functional.one_hot(train_target, 2).float()
        test_target_hot = nn.functional.one_hot(test_target, 2).float()
            
    if setup == 'AuxiliarySetup':
        # Split pairs (flatten) into individual pictures
        train_input = train_input_original.reshape((2*N, 1, 14, 14))
        test_input = test_input_original.reshape((2*N, 1, 14, 14))
        # Flatten targets accordingly
        train_target = train_aux_target.reshape(2*N)
        test_target = test_aux_target.reshape(2*N)
        # Hot encoding
        train_target_hot = nn.functional.one_hot(train_target, 10).float()
        test_target_hot = nn.functional.one_hot(test_target, 10).float()
            
    batch_size = 100
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) 
    criterion = nn.CrossEntropyLoss() if use_crossentropy else nn.MSELoss()
    
    # If we use cross-entropy, we don't use hot-encoded targets
    trainDataset = TrainDataset(train_input, 
                                train_target if use_crossentropy else train_target_hot,
                                data_augment)
    dataloader = DataLoader(trainDataset, 
                            batch_size=batch_size, shuffle=True, num_workers=0)
    train_errors = []
    test_errors = []
    
    for e in range(epochs):
        if verbose :
            print('.', end='')
        for i_batch, batch in enumerate(dataloader):
            # Forward pass
            output = model(batch['input'])
            # Compute loss
            loss = criterion(output, batch['target'])
            # Backprop & update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        with torch.no_grad():
            
            if setup == 'PairSetup':
                # Compute train error
                output = model(train_input)
                train_errors.append(compute_error(output, train_target_hot))
                # Compute test error
                output = model(test_input)
                test_errors.append(compute_error(output, test_target_hot))
            
            # If auxiliary setup, we recognize each picture of the pair individualy and then
            # deduce which digit is lesser or equal to the other
            if setup == 'AuxiliarySetup':
                # Compute train error
                output_digit1 = model(train_input_original[:,0:1,:,:]).argmax(dim=1)
                output_digit2 = model(train_input_original[:,1:2,:,:]).argmax(dim=1)
                final_prediction = (output_digit1 <= output_digit2).long()
                train_errors.append(compute_error_class(final_prediction, train_pair_target))
                # Compute test error
                output_digit1 = model(test_input_original[:,0:1,:,:]).argmax(dim=1)
                output_digit2 = model(test_input_original[:,1:2,:,:]).argmax(dim=1)
                final_prediction = (output_digit1 <= output_digit2).long()
                test_errors.append(compute_error_class(final_prediction, test_pair_target))
                           
    return train_errors, test_errors


def rounds_train(model, setup, rounds=10, augment_data=False, use_crossentropy=True, 
                          lr=1e-3, epochs=200, verbose=False, plot_title = None,
                          plot_file_path = None, data_augment=False):
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
        the given title will be saved at the given path
        - data_augment: bool, if True pairs will be augmented by inversing them. Only works for PairSetup.
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
        train_input, train_pair_target, train_aux_target , test_input, test_pair_target, test_aux_target = prologue.generate_pair_sets(N)
        # Train
        train_errors, test_errors = train(model,
                                          setup,
                                          train_input, train_pair_target, train_aux_target,
                                          test_input, test_pair_target, test_aux_target,
                                          use_crossentropy=use_crossentropy, lr=lr, epochs=epochs,
                                          verbose=False,
                                          data_augment = data_augment)
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