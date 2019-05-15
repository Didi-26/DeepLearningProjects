import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import dlc_practical_prologue as prologue
import numpy as np

# Our code is contained in the following modules :
import models  # contains all our torch models classes
import plots  # custom ploting functions to produce figures of the report
import training_functions  # all our functions and classes for training

msg = '''\n/!\ Running the script with all experiments enabled will produce all
    the figure & results of the report but will take an extremely long
    time to compute. To produce only certain figures, set the experiments
    you wish to run to True and those to skip to False at the top of this
    script.\n\n'''
print(msg)

# Set here which experiment is to be run (running them all will take a long
# time to compute)
run_LossCompare = True
# Experiments for "PairSetup" : In this setup, we consider directly the pairs
# as input to the network. Thus our inputs are N samples of [2,14,14] made of
# two 14*14 pictures. Our ouputs are the class 0 or 1  indicating whereas if
# the first digit is lesser or equal to the second.
run_PairSetup_SimpleLinear = True
run_PairSetup_MLP = True
run_PairSetup_LeNetLike5 = True
run_PairSetup_LeNetLike3 = True
run_PairSetup_VGGNetLike = True
run_PairSetup_ResNet = True
# Experiments for "AuxiliarySetup" : In this setup, we consider N individual
# 14*14 pictures as input. The network use an auxiliary loss to learn to
# classify those from 0 to 9. The auxiliary outputs are the the class 0 to 9
# corresponding to the digit on the picture. We then use this network to
# predict the number and we can then do the difference to perform our original
# goal which is to predict whereas if the first digit is lesser or equal to the
# second
run_AuxiliarySetup_SimpleLinear = True
run_AuxiliarySetup_MLP = True
run_AuxiliarySetup_LeNetLike5 = True
run_AuxiliarySetup_LeNetLike3 = True
run_AuxiliarySetup_VGGNetLike = True
run_AuxiliarySetup_ResNet = True
# Comparaison perf with /without data-augmentation
run_PairSetup_MLP_DataAugmented_Compare = True

# Set a fixed seed for reproducibility
random_seed = 42


def count_parameters(model):
    """ Returns the number of trainable parameters of the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if run_LossCompare:
    print('******************** Running Loss comparaison ********************')

    print('--- PairSetup, MSE loss : ---')
    torch.manual_seed(random_seed)
    in_dim, out_dim = 14 * 14 * 2, 2
    model = models.SimpleLinear(in_dim, out_dim)
    test_err_mean, test_err_std, _, _ = training_functions.rounds_train(
        model, 'PairSetup', lr=0.00004, epochs=300, use_crossentropy=False)
    print('mean minimum test error : {0:.{1}f} %'.format(test_err_mean, 1))
    print('std minimum test error : {0:.{1}f} %'.format(test_err_std, 1))

    print('--- PairSetup, cross-entropy loss : ---')
    torch.manual_seed(random_seed)
    model = models.SimpleLinear(in_dim, out_dim)
    test_err_mean, test_err_std, _, _ = training_functions.rounds_train(
        model, 'PairSetup', lr=0.00004, epochs=300, use_crossentropy=True)
    print('mean minimum test error : {0:.{1}f} %'.format(test_err_mean, 1))
    print('std minimum test error : {0:.{1}f} %'.format(test_err_std, 1))

    print('--- AuxiliarySetup, MSE loss : ---')
    torch.manual_seed(random_seed)
    in_dim, out_dim = 14 * 14, 10
    model = models.SimpleLinear(in_dim, out_dim)
    test_err_mean, test_err_std, _, _ = training_functions.rounds_train(
        model,
        'AuxiliarySetup',
        lr=0.00004,
        epochs=300,
        use_crossentropy=False)
    print('mean minimum test error : {0:.{1}f} %'.format(test_err_mean, 1))
    print('std minimum test error : {0:.{1}f} %'.format(test_err_std, 1))

    print('--- AuxiliarySetup, cross-entropy loss : ---')
    torch.manual_seed(random_seed)
    model = models.SimpleLinear(in_dim, out_dim)
    test_err_mean, test_err_std, _, _ = training_functions.rounds_train(
        model, 'AuxiliarySetup', lr=0.00004, epochs=300, use_crossentropy=True)
    print('mean minimum test error : {0:.{1}f} %'.format(test_err_mean, 1))
    print('std minimum test error : {0:.{1}f} %'.format(test_err_std, 1))

if run_PairSetup_SimpleLinear:
    torch.manual_seed(random_seed)
    print('********* Running SimpleLinear model (for PairSetup) *************')
    in_dim, out_dim = 14 * 14 * 2, 2
    model = models.SimpleLinear(in_dim, out_dim)
    test_err_mean, test_err_std, _, _ = training_functions.rounds_train(
        model,
        'PairSetup',
        plot_title='Pair Setup Linear Classifier error history',
        plot_file_path='./plots/pairSetup_SimpleLinear.svg',
        lr=0.00004,
        epochs=300,
        use_crossentropy=True,
        verbose=True,
        rounds=10)
    print('mean minimum test error : {0:.{1}f} %'.format(test_err_mean, 1))
    print('std minimum test error : {0:.{1}f} %'.format(test_err_std, 1))
    print('number of trainable parameters : {}'.format(
        count_parameters(model)))

if run_PairSetup_MLP:
    torch.manual_seed(random_seed)
    print('************** Running MLP model (for PairSetup) *****************')
    in_dim, out_dim = 14 * 14 * 2, 2
    # We test for different hidden layer count vs. neurons per layer count
    L_range = list(range(2, 18, 2))
    h_range = list(range(10, 50, 5))
    test_error_means = np.zeros((len(h_range), len(L_range)))
    test_error_std = np.zeros((len(h_range), len(L_range)))
    for L_idx in range(0, len(L_range)):
        for h_idx in range(0, len(h_range)):
            torch.manual_seed(random_seed)
            L = L_range[L_idx]
            h = h_range[h_idx]
            print('testing with L = {} and h = {}'.format(L, h))
            model = models.MLP(L, h, in_dim, out_dim)
            test_err_mean, test_err_std, _, _ = training_functions.rounds_train(
                model,
                'PairSetup',
                lr=0.0001,
                epochs=200,
                use_crossentropy=True,
                rounds=10)
            test_error_means[h_idx, L_idx] = test_err_mean
            test_error_std[h_idx, L_idx] = test_err_std
    # Plot heat table
    plots.plot_error_table(
        h_range, L_range, test_error_means, test_error_std,
        'Pair Setup MLP models mean/std minimum test error',
        './plots/pairSetup_MLP.svg')

if run_PairSetup_LeNetLike5:
    torch.manual_seed(random_seed)
    print('*********** Running LetNetLike5 model (for PairSetup) ************')
    in_depth, out_dim = 2, 2
    model = models.LeNetLike5(in_depth, out_dim)
    test_err_mean, test_err_std, _, _ = training_functions.rounds_train(
        model,
        'PairSetup',
        plot_title='Pair Setup LeNetLike5 error history',
        plot_file_path='./plots/pairSetup_LeNetLike5.svg',
        lr=0.0001,
        epochs=300,
        use_crossentropy=True,
        verbose=True,
        rounds=10)
    print('mean minimum test error : {0:.{1}f} %'.format(test_err_mean, 1))
    print('std minimum test error : {0:.{1}f} %'.format(test_err_std, 1))
    print('number of trainable parameters : {}'.format(
        count_parameters(model)))

if run_PairSetup_LeNetLike3:
    torch.manual_seed(random_seed)
    print('********** Running LetNetLike3 model (for PairSetup) *************')
    in_depth, out_dim = 2, 2
    model = models.LeNetLike3(in_depth, out_dim)
    test_err_mean, test_err_std, _, _ = training_functions.rounds_train(
        model,
        'PairSetup',
        plot_title='Pair Setup LeNetLike3 error history',
        plot_file_path='./plots/pairSetup_LeNetLike3.svg',
        lr=0.00015,
        epochs=300,
        use_crossentropy=True,
        verbose=True,
        rounds=10)
    print('mean minimum test error : {0:.{1}f} %'.format(test_err_mean, 1))
    print('std minimum test error : {0:.{1}f} %'.format(test_err_std, 1))
    print('number of trainable parameters : {}'.format(
        count_parameters(model)))

if run_PairSetup_VGGNetLike:
    torch.manual_seed(random_seed)
    print('********** Running VGGNetLike model (for PairSetup) **************')
    in_depth, out_dim = 2, 2
    model = models.VGGNetLike(in_depth, out_dim)
    test_err_mean, test_err_std, _, _ = training_functions.rounds_train(
        model,
        'PairSetup',
        plot_title='Pair Setup VGGNetLike error history',
        plot_file_path='./plots/pairSetup_VGGNetLike.svg',
        lr=0.0001,
        epochs=300,
        use_crossentropy=True,
        verbose=True,
        rounds=10)
    print('mean minimum test error : {0:.{1}f} %'.format(test_err_mean, 1))
    print('std minimum test error : {0:.{1}f} %'.format(test_err_std, 1))
    print('number of trainable parameters : {}'.format(
        count_parameters(model)))

if run_PairSetup_ResNet:
    torch.manual_seed(random_seed)
    print('************ Running ResNet model (for PairSetup) ****************')
    in_depth, out_dim = 2, 2
    model = models.ResNet(in_depth, out_dim, 8)
    test_err_mean, test_err_std, _, _ = training_functions.rounds_train(
        model,
        'PairSetup',
        plot_title='Pair Setup ResNet error history',
        plot_file_path='./plots/pairSetup_ResNet.svg',
        lr=0.0002,
        epochs=300,
        use_crossentropy=True,
        verbose=True,
        rounds=10)
    print('mean minimum test error : {0:.{1}f} %'.format(test_err_mean, 1))
    print('std minimum test error : {0:.{1}f} %'.format(test_err_std, 1))
    print('number of trainable parameters : {}'.format(
        count_parameters(model)))

if run_AuxiliarySetup_SimpleLinear:
    torch.manual_seed(random_seed)
    print('******** Running SimpleLinear model (for AuxiliarySetup) *********')
    in_dim, out_dim = 14 * 14, 10
    model = models.SimpleLinear(in_dim, out_dim)
    test_err_mean, test_err_std, _, _ = training_functions.rounds_train(
        model,
        'AuxiliarySetup',
        plot_title='Auxiliary Setup Linear Classifier error history',
        plot_file_path='./plots/auxiliarySetup_SimpleLinear.svg',
        lr=0.00004,
        epochs=300,
        use_crossentropy=True,
        verbose=True,
        rounds=10)
    print('mean minimum test error : {0:.{1}f} %'.format(test_err_mean, 1))
    print('std minimum test error : {0:.{1}f} %'.format(test_err_std, 1))
    print('number of trainable parameters : {}'.format(
        count_parameters(model)))

if run_AuxiliarySetup_MLP:
    torch.manual_seed(random_seed)
    print('*********** Running MLP model (for AuxiliarySetup) ***************')
    in_dim, out_dim = 14 * 14, 10
    # We test for different hidden layer count vs. neurons per layer count
    L_range = list(range(2, 18, 2))
    h_range = list(range(10, 50, 5))
    test_error_means = np.zeros((len(h_range), len(L_range)))
    test_error_std = np.zeros((len(h_range), len(L_range)))
    for L_idx in range(0, len(L_range)):
        for h_idx in range(0, len(h_range)):
            torch.manual_seed(random_seed)
            L = L_range[L_idx]
            h = h_range[h_idx]
            print('testing with L = {} and h = {}'.format(L, h))
            model = models.MLP(L, h, in_dim, out_dim)
            test_err_mean, test_err_std, _, _ = training_functions.rounds_train(
                model,
                'AuxiliarySetup',
                lr=0.00015,
                epochs=200,
                use_crossentropy=True,
                rounds=10)
            test_error_means[h_idx, L_idx] = test_err_mean
            test_error_std[h_idx, L_idx] = test_err_std
    # Plot heat table
    plots.plot_error_table(
        h_range, L_range, test_error_means, test_error_std,
        'Auxiliary Setup MLP models mean/std minimum test error',
        './plots/auxiliarySetup_MLP.svg')

if run_AuxiliarySetup_LeNetLike5:
    torch.manual_seed(random_seed)
    print('******* Running LetNetLike5 model (for AuxiliarySetup) ***********')
    in_depth, out_dim = 1, 10
    model = models.LeNetLike5(in_depth, out_dim)
    test_err_mean, test_err_std, _, _ = training_functions.rounds_train(
        model,
        'AuxiliarySetup',
        plot_title='Auxiliary Setup LeNetLike5 error history',
        plot_file_path='./plots/auxiliarySetup_LeNetLike5.svg',
        lr=0.0001,
        epochs=300,
        use_crossentropy=True,
        verbose=True,
        rounds=10)
    print('mean minimum test error : {0:.{1}f} %'.format(test_err_mean, 1))
    print('std minimum test error : {0:.{1}f} %'.format(test_err_std, 1))
    print('number of trainable parameters : {}'.format(
        count_parameters(model)))

if run_AuxiliarySetup_LeNetLike3:
    torch.manual_seed(random_seed)
    print('******** Running LetNetLike3 model (for AuxiliarySetup) **********')
    in_depth, out_dim = 1, 10
    model = models.LeNetLike3(in_depth, out_dim)
    test_err_mean, test_err_std, _, _ = training_functions.rounds_train(
        model,
        'AuxiliarySetup',
        plot_title='Auxiliary Setup LeNetLike3 error history',
        plot_file_path='./plots/auxiliarySetup_LeNetLike3.svg',
        lr=0.0001,
        epochs=300,
        use_crossentropy=True,
        verbose=True,
        rounds=10)
    print('mean minimum test error : {0:.{1}f} %'.format(test_err_mean, 1))
    print('std minimum test error : {0:.{1}f} %'.format(test_err_std, 1))
    print('number of trainable parameters : {}'.format(
        count_parameters(model)))

if run_AuxiliarySetup_VGGNetLike:
    torch.manual_seed(random_seed)
    print('******** Running VGGNetLike model (for AuxiliarySetup) ***********')
    in_depth, out_dim = 1, 10
    model = models.VGGNetLike(in_depth, out_dim)
    test_err_mean, test_err_std, _, _ = training_functions.rounds_train(
        model,
        'AuxiliarySetup',
        plot_title='Auxiliary Setup VGGNetLike error history',
        plot_file_path='./plots/auxiliarySetup_VGGNetLike.svg',
        lr=0.00025,
        epochs=300,
        use_crossentropy=True,
        verbose=True,
        rounds=10)
    print('mean minimum test error : {0:.{1}f} %'.format(test_err_mean, 1))
    print('std minimum test error : {0:.{1}f} %'.format(test_err_std, 1))
    print('number of trainable parameters : {}'.format(
        count_parameters(model)))

if run_AuxiliarySetup_ResNet:
    torch.manual_seed(random_seed)
    print('******** Running ResNet model (for AuxiliarySetup) ***************')
    in_depth, out_dim = 1, 10
    model = models.ResNet(in_depth, out_dim, 8)
    test_err_mean, test_err_std, _, _ = training_functions.rounds_train(
        model,
        'AuxiliarySetup',
        plot_title='Auxiliary Setup ResNet error history',
        plot_file_path='./plots/auxiliarySetup_ResNet.svg',
        lr=0.0004,
        epochs=300,
        use_crossentropy=True,
        verbose=True,
        rounds=10)
    print('mean minimum test error : {0:.{1}f} %'.format(test_err_mean, 1))
    print('std minimum test error : {0:.{1}f} %'.format(test_err_std, 1))
    print('number of trainable parameters : {}'.format(
        count_parameters(model)))

if run_PairSetup_MLP_DataAugmented_Compare:
    torch.manual_seed(random_seed)
    print('*************** Data-Augmentation Comparaison ********************')
    print('******* Running DataAugmented MLP model (for PairSetup) **********')
    in_dim, out_dim = 14 * 14 * 2, 2
    model = models.MLP(10, 45, in_dim, out_dim)
    test_err_mean, test_err_std, _, _ = training_functions.rounds_train(
                model,
                'PairSetup',
                lr=0.00015,
                epochs=400,
                use_crossentropy=True,
                rounds=10,
                verbose=True,
                data_augment=True)
    print('mean minimum test error : {0:.{1}f} %'.format(test_err_mean, 1))
    print('std minimum test error : {0:.{1}f} %'.format(test_err_std, 1))
    print('number of trainable parameters : {}'.format(
        count_parameters(model)))

    torch.manual_seed(random_seed)
    print('***** Running Non-DataAugmented MLP model (for PairSetup) ********')
    in_dim, out_dim = 14 * 14 * 2, 2
    model = models.MLP(10, 50, in_dim, out_dim)
    test_err_mean, test_err_std, _, _ = training_functions.rounds_train(
                model,
                'PairSetup',
                lr=0.00015,
                epochs=400,
                use_crossentropy=True,
                rounds=10,
                verbose=True)
    print('mean minimum test error : {0:.{1}f} %'.format(test_err_mean, 1))
    print('std minimum test error : {0:.{1}f} %'.format(test_err_std, 1))
    print('number of trainable parameters : {}'.format(
        count_parameters(model)))
