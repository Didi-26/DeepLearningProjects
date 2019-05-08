import torch # NB: torch is only imported for comparison
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
import framework
import matplotlib.pyplot as plt
import math

########################### Useful functions ############################
def generate_disc_set(nb):
    """
    Function to generate a dataset using a uniform distribution in [0,1]^2
    With label=1 inside the disk of radius 1/sqrt(2*pi) and 0 outside of it
    Argument: nb (integer) number of samples
    Returns: 
        - torch tensor of dimensions (nb,2), point coordinates
        - torch tensor of dimensions (nb,2), hot-encoded point labels
    """
    # generate unif samples
    p = torch.FloatTensor(nb, 2).uniform_(0, 1)
    # assign label inside/outside circle
    center = torch.tensor([0.5,0.5]).float()
    inside = ( torch.norm(p-center, dim=1) < (1.0 / math.sqrt(2*math.pi)) )
    # Hot encoding
    label = torch.zeros(inside.shape[0], 2)
    label[inside==1,1] = 1
    label[inside==0,0] = 1
    return p, label

def compute_error_percent(prediction, target) :
    """
    Function to compute classification error in %
    Arguments: 
        - prediction: torch tensor, usually corresponds to the output of the model
        - traget: torch tensor of the same size, corresponds to ground truth
    Returns: 
        - torch tensor that contains error in percent for each sample
    """
    errors = (prediction.argmax(dim=1) != target.argmax(dim=1))
    return (100 * errors.sum() / target.shape[0]).item()

def compute_test_error(model, test_input, test_target) :
    """
    Function to compute classification error for test set
    Arguments: 
        - model: the classification model
        - test_input: torch tensor, test dataset 
        - test_target: torch tensor, test labels
    Returns: 
        - torch tensor that contains classification error in % for test set
    """
    # Forward pass
    output = model.forward(test_input)
    # Compute error %
    return compute_error_percent(output, test_target)

def train(model, optimizer,  x_train, y_train) :
    """
    Function that trains a model. 
    Arguments: 
        - model: model to train
        - optimizer: framework.Optimizer object, the chosen optimizer for training
        - x_train: torch tensor, training dataset
        - y_train: torch tensor, training labels
    Returns: 
        - list, training loss over epochs
        - list, training error in % over epochs
        - list, test error in % over epochs
    """
    mini_batch_size = 200
    epochs = 300
    train_loss_hist = []
    train_error_hist = []
    test_error_hist = []
    for e in range(0, epochs):
        # variables to accumulate loss and error over the batches of this epoch
        train_loss_buffer = 0
        train_error_buffer = 0
        for b in range(0, x_train.size(0), mini_batch_size):
                # Forward pass of neural net
                x_batch = x_train.narrow(0, b, mini_batch_size)
                y_batch = y_train.narrow(0, b, mini_batch_size)
                output = model.forward(x_batch)
                # Compute loss
                loss_module = framework.LossMSE(y_batch)
                loss = loss_module.forward(output)
                train_loss_buffer += loss.item()
                # Compute train error
                train_error_buffer += compute_error_percent(output,y_batch)
                # Set all gradients to zero
                optimizer.zero_grad()
                # Backward
                model.backward(loss_module.backward())
                # Step
                optimizer.step()
        # Store average loss & error for this epoch
        batch_per_epoch = x_train.size(0) / mini_batch_size
        train_loss_hist.append(train_loss_buffer / batch_per_epoch)
        train_error_hist.append(train_error_buffer / batch_per_epoch)
        # Compute a test error
        test_error_hist.append(compute_test_error(model, x_test, y_test))
    return train_loss_hist, train_error_hist, test_error_hist

def define_model(H, hidden_nb, d_in, d_out):
    """
    Function that defines a fully connected network with H hidden layers.
    Arguments:
        - H: int, number of units in the fully connected layers
        - hidden_nb: int, number of hidden layers
        - d_in: int, input dimension
        - d_out: int, output dimension
    Returns: 
        - Fully connected network with H hidden layers
    """
    modules_array = []
    # Input layer
    modules_array.append(framework.Linear(d_in, H, 'relu'))
    # Hidden layers
    for i in range(0, hidden_nb):
        modules_array.append(framework.ReLU())
        modules_array.append(framework.Linear(H, H, 'relu'))
    # Output layer
    modules_array.append(framework.ReLU())
    modules_array.append(framework.Linear(H, d_out, 'none'))
    model = framework.Sequential(modules_array)
    return model
##############################################################################
    
print("Testing the implementation of the framework...")

print("Testing the computation of loss...")
loss_error = []
for n in range(100):
    torch.manual_seed(n)
    y_output = torch.randn((10,2)).type(torch.double)
    torch.manual_seed(n+1)
    y_target = torch.randn((10,2)).type(torch.double)
    loss_module = framework.LossMSE(y_target)
    our_loss = loss_module.forward(y_output).type(torch.double)
    loss_mod_torch = nn.MSELoss()
    pytorch_loss = loss_mod_torch(y_output, y_target).type(torch.double)
    loss_error.append(abs(our_loss.item() - pytorch_loss.item()))
    
assert(torch.Tensor(loss_error).max().item() <= 5e-15),\
 "Test of computation of loss... Failed"
print("Test of computation of loss... Passed")

print("Testing forward pass...")

# Create a fully connected network using the framework
H = 25
hidden_nb = 1
d_in = 2
d_out = 1
model = define_model(H, hidden_nb, d_in, d_out)

# Create a similar network using PyTorch 
torch_model = nn.Sequential(
      nn.Linear(2, H, bias=True),
      nn.ReLU(),
      nn.Linear(H, H, bias=True),
      nn.ReLU(),
      nn.Linear(H, 1),
      )

# Initialize the PyTorch network using the same weights as the framework net
w0 = model.param()[0][0][0]
b0 = model.param()[0][1][0]
w1 = model.param()[2][0][0]
b1 = model.param()[2][1][0]
w2 = model.param()[4][0][0]
b2 = model.param()[4][1][0]
torch_model[0].weight.data = w0
torch_model[0].bias.data = b0
torch_model[2].weight.data = w1
torch_model[2].bias.data = b1
torch_model[4].weight.data = w2
torch_model[4].bias.data = b2

# Define a random input 
torch.manual_seed(678) # For reproductibility
input_set = torch.randn(100,2)

# Compute outputs of forward pass for both networks
output = model.forward(input_set)
output_torch = torch_model(input_set)

assert((output - output_torch).norm().item() <= 1e-17), \
    "Test of forward pass... Failed"
print("Test of forward pass... Passed")

print("Test of backward pass...")

# Create a target variable model_torch
torch.manual_seed(345)
target = Variable(torch.randn(100,1))

#Define loss for the first network
loss_module = framework.LossMSE(target)
our_loss = loss_module.forward(output)


# Compute gradient w.r.t input for the first network
grad = model.backward(loss_module.backward())

# Create the PyTorch network again, with autograd enabled
torch.set_grad_enabled(True)
torch_model = nn.Sequential(
      nn.Linear(2, H, bias=True),
      nn.ReLU(),
      nn.Linear(H, H, bias=True),
      nn.ReLU(),
      nn.Linear(H, 1),
      )

# Initialize the second network like the first
torch_model[0].weight.data = w0
torch_model[0].bias.data = b0
torch_model[2].weight.data = w1
torch_model[2].bias.data = b1
torch_model[4].weight.data = w2
torch_model[4].bias.data = b2

# Define variables
input_set_torch = Variable(input_set, requires_grad=True)
target_torch = Variable(target, requires_grad=True)

# Compute output 
output_torch = torch_model(input_set_torch)

# Define loss
loss_mod_torch = nn.MSELoss()
pytorch_loss = loss_mod_torch(output_torch,target_torch)
pytorch_loss.backward()

# Compute gradient w.r.t output using autograd
grad_torch = input_set_torch.grad

# Disable autograd 
torch.set_grad_enabled(False)

assert((grad - grad_torch).norm().item() < 1e-17), \
    "Test of backward pass... Failed"
print("Test of backward pass... Passed")

# Test on a simple example 
print("Test the framework on a simple example...")

print("Generate data...")
N = 1000
torch.manual_seed(4567) #For reproductibility
x_train, y_train = generate_disc_set(N)
torch.manual_seed(9086) #For reproductibility
x_test, y_test = generate_disc_set(N)

print("Saving training set plot...")
plt.figure(figsize=(6,6))
plt.scatter(x_train[y_train[:,1] ==1][:,0], x_train[y_train[:,1] ==1][:,1])
plt.scatter(x_train[y_train[:,0] ==1][:,0], x_train[y_train[:,0] ==1][:,1])
plt.title('Training dataset', fontsize=20)
plt.savefig('train_set.eps', bbox_inches='tight')

print("Train a model using a SGD optimizer...")
H = 25 # hidden layer size
hidden_nb = 3 # hidden layers number
d_in = 2
d_out = 2
model = define_model(H, hidden_nb, d_in, d_out)

# Define SGD optimizers
lr = 1e-2
optimizer_sgd = framework.SGD(model.param(), lr)

# Training the model using the SGD optimizer
train_loss_hist, train_error_hist, test_error_hist = train(model, optimizer_sgd, x_train, y_train)

print("Train re-initialized model using Adam optimizer...")
# Re-initialize the model
model = define_model(H, hidden_nb, d_in, d_out)

# Define an Adam optimizer
optimizer_adam = framework.Adam(model.param(), lr)

# Training the model using the Adam optimizer
train_loss_hist_a, train_error_hist_a, test_error_hist_a = train(model, optimizer_adam, x_train, y_train)

print("Save plots...")

plt.figure(figsize=(10,5))
plt.plot(train_loss_hist,label='SGD')
plt.plot(train_loss_hist_a,label='Adam')
plt.ylabel('MSE loss')
plt.xlabel('Epoch')
plt.title('MSE loss using different optimizers', fontsize=18)
plt.grid(True)
plt.legend()
plt.savefig('compare_optimizers.eps', bbox_inches='tight')
plt.show()

plt.figure(figsize=(10,5))
plt.plot(train_error_hist_a,label='train error %')
plt.plot(test_error_hist_a, label='test error %')
plt.ylabel('Error %')
plt.xlabel('Epoch')
plt.title('Train and test error using an Adam optimizer')
plt.grid(True)
plt.legend()
plt.savefig('train_test_error_Adam.eps', bbox_inches='tight')
plt.show()

plt.figure(figsize=(10,5))
plt.plot(train_error_hist,label='train error %')
plt.plot(test_error_hist, label='test error %')
plt.ylabel('Error %')
plt.xlabel('Epoch')
plt.title('Train and test error using an SGD optimizer')
plt.grid(True)
plt.legend()
plt.savefig('train_test_error_SGD.eps', bbox_inches='tight')
plt.show()

print("All tests passed !")