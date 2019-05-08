import matplotlib.pyplot as plt

def plot_errors(train_errors, test_errors, title, file_path):
    """
    Plots on the same graph multiple traces of train error history and
    multiple traces of test error history
    Arguments:
        - train_errors: list of train error histories
        - test_errors: list of test error histories
        - file_path: will save the figure to this complete path (with 
        filename and extension)
    """
    plt.figure(figsize=(10,5))
    plt.title(title)
    plt.xlabel('epoch')
    plt.ylabel('error %')
    plt.plot(np.array(train_errors).T, c='b', label='train error %')
    plt.plot(np.array(test_errors).T, c='o', label='test error %')
    plt.ylim(0, 60)
    plt.grid()
    
    handles, labels = plt.gca().get_legend_handles_labels()
    labels, ids = np.unique(labels, return_index=True)
    handles = [handles[i] for i in ids]
    plt.legend(handles, labels, loc='best')

    plt.savefig(file_path, bbox_inches='tight')