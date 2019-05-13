import matplotlib.pyplot as plt
import numpy as np


def plot_errors(train_errors, test_errors, title, file_path):
    """
    Plots on the same graph multiple traces of train error history and
    multiple traces of test error history. Also plot in a stronger color
    the average of the traces.
    Arguments:
        - train_errors: list of train error histories
        - test_errors: list of test error histories
        - file_path: will save the figure to this complete path (with
        filename and extension)
    """
    train_histories = np.array(train_errors)
    test_histories = np.array(test_errors)

    plt.figure(figsize=(10, 5))
    plt.title(title)
    plt.xlabel('epoch')
    plt.ylabel('error %')
    # Plot in half transparent all the traces
    plt.plot(train_histories.T, c='#1400f028')
    plt.plot(test_histories.T, c='#fa000028')
    # Plot in plain color an average of the traces
    plt.plot(np.mean(train_histories, axis=0).T,
             c='#1400f0ff',
             label='train error %')
    plt.plot(np.mean(test_histories, axis=0),
             c='#fa0000ff',
             label='test error %')
    plt.ylim(0, 60)
    plt.grid()
    plt.legend()

    plt.savefig(file_path, bbox_inches='tight')


def plot_error_table(h_range, L_range, table_mean, table_std, title,
                     file_path):
    """
    Plots a heat table of h vs L. Displays the mean and the std in each cell.
    Arguments:
        - h_range: list values of h
        - L_range: list of values of L
        - table_mean: mean error values table
        - table_std: std error values table
    """
    h_labels = ['h = {}'.format(h) for h in h_range]
    L_labels = ['L = {}'.format(L) for L in L_range]

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(table_mean)

    ax.set_xticks(np.arange(len(L_labels)))
    ax.set_yticks(np.arange(len(h_labels)))
    ax.set_xticklabels(L_labels)
    ax.set_yticklabels(h_labels)

    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(),
             rotation=45,
             ha='right',
             rotation_mode='anchor')

    # Loop over data dimensions and create text annotations.
    for i in range(len(h_labels)):
        for j in range(len(L_labels)):
            text = ax.text(j,
                           i,
                           '{0:.{1}f}'.format(table_mean[i, j], 1),
                           ha='center',
                           va='bottom',
                           color='w')
            text = ax.text(j,
                           i,
                           '({0:.{1}f})'.format(table_std[i, j], 1),
                           ha='center',
                           va='top',
                           color='#FFFFFF80')

    ax.set_title(title)
    fig.tight_layout()
    plt.savefig(file_path, bbox_inches='tight')
