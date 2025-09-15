import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
#Step 1: Training data generating

def dataset(show=True):
    x1 = np.arange(-500, 500, 1)
    x2 = np.arange(-500, 500, 1)
    x3 = np.arange(-500, 500, 1)
 
    # 產生 y
    y = x1 - x2**3 + 100*x3 + 4 + np.random.randn(len(x1)) * 1000
    # 合併 x
    X = np.stack([x1, x2, x3], axis=1)
    if show:
        plt.figure(figsize=(8,6))
        plt.scatter(x1, y, s=1, alpha=0.3)
        plt.xlabel('x1')
        plt.ylabel('y')
        plt.title('x1 vs y')
        plt.show()
    return X, y

# Call the 'dataset' function and assign the returned values to x and y
X, y = dataset()

print("Shape of X:", X.shape)  # X 是 (N, 3)
print("Shape of y:", y.shape)
print('Type of X: ', type(X))
print('Type of y: ', type(y))
print('min(X[:,0]), max(X[:,0]): ', min(X[:,0]), max(X[:,0]))
print('min(X[:,1]), max(X[:,1]): ', min(X[:,1]), max(X[:,1]))
print('min(X[:,2]), max(X[:,2]): ', min(X[:,2]), max(X[:,2]))
print('min(y), max(y): ', min(y), max(y))

# Normalize 每個維度
X = X / np.max(np.abs(X), axis=0)
y = y / np.max(np.abs(y))

print('min(X[:,0]), max(X[:,0]): ', min(X[:,0]), max(X[:,0]))
print('min(X[:,1]), max(X[:,1]): ', min(X[:,1]), max(X[:,1]))
print('min(X[:,2]), max(X[:,2]): ', min(X[:,2]), max(X[:,2]))
print('min(y), max(y): ', min(y), max(y))
# Convert the 'x' array to a PyTorch tensor with float data type and add a new dimension (unsqueeze) along axis 1
X_torch = torch.tensor(X, dtype=torch.float).unsqueeze(1)

# Convert the 'y' array to a PyTorch tensor with float data type and add a new dimension (unsqueeze) along axis 1
y_torch = torch.tensor(y, dtype=torch.float).unsqueeze(1)
print('Type of X: ',type(X_torch))
print('Type of y: ',type(y_torch))
print(X_torch.shape)
print(y_torch.shape)
## Step 1: Define the model architecture
# Create a simple linear regression model using a single linear layer (in_features=1, out_features=1)
model = torch.nn.Sequential(torch.nn.Linear(in_features=3, out_features=1))

## Step 2: Set Hyperparameter
# Define the learning rate for the optimizer
learning_rate = 1e-3

## Step 3: Define loss function
# Define the mean squared error (MSE) loss function for regression tasks
loss_func = torch.nn.MSELoss(reduction='mean')

## Step 4: Define the optimizer
# Use the Adam optimizer to update the model's parameters with the specified learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

## Step 5: Define the number of training epochs

num_ep = 1000

## Step 6: Declare an empty list for losses. This will keep track of all losses during training
loss_history = []

## Step 7: Training loop
# Loop over the specified number of training epochs
for ep in range(num_ep):
    # Forward pass: Predict the output (y_pred) using the current model
    y_pred = model(X_torch)

    # Calculate the MSE loss between predicted values and actual values
    loss = loss_func(y_pred, y_torch)

    # Append the current loss to the loss history list
    loss_history.append(loss.item())

    # Zero the gradients to prevent accumulation
    optimizer.zero_grad()

    # Backpropagate the gradients (compute gradients with respect to model parameters)
    loss.backward()

    # Update the model's parameters using the optimizer
    optimizer.step()

    # Plot the loss history over training epochs (optional)
    plt.plot(loss_history)

## Step 8: Plot the loss history
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.show()

## Step 9: Compute the final MSE and make prediction
mse = loss_history[-1]

# Make predictions on the input data using the trained model
y_hat = model(X_torch).detach().numpy()

## Step 10: Visualize the original data, predicted line, and MSE value
plt.figure(figsize=(12, 7))
plt.title('PyTorch Model')
plt.scatter(X[:, 0], y, label='Data $(x1, y)$')
plt.plot(X[:, 0], y_hat.squeeze(), color='red', label='Predicted Line $y = f(x)$', linewidth=4.0)
plt.xlabel('$x$', fontsize=20)
plt.ylabel('$y$', fontsize=20)
plt.text(0, 0.70, 'MSE = {:.3f}'.format(mse), fontsize=20)
plt.grid(True)
plt.legend(fontsize=20)
plt.show()
# Define a custom neural network class named 'Net_no_activation' without activation functions
class Net_no_activation(nn.Module):
    def __init__(self, hidden_size):
        super(Net_no_activation, self).__init__()
        self.hidden_size = hidden_size
        self.layer1 = nn.Linear(3, self.hidden_size)  # 輸入維度改成3
        self.layer2 = nn.Linear(self.hidden_size, 1)  # 輸出維度不變

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

# Create an instance of the 'Net_no_activation' neural network with a hidden size of 100
multi_layer_model_no_activation = Net_no_activation(hidden_size=100)

# Set the learning rate for the optimizer
learning_rate = 1e-3

# Define the mean squared error (MSE) loss function for regression tasks
loss_func = torch.nn.MSELoss(reduction='mean')

# Initialize an Adam optimizer for the model's parameters
multi_layer_optimizer_no_activation = torch.optim.Adam(multi_layer_model_no_activation.parameters(), lr=learning_rate)

# Specify the number of training epochs
num_ep = 500

# Create an empty list to store loss values during training
loss_history = []

# Training loop
for ep in range(num_ep):
    # Forward pass: Predict the output (y_pred) using the current model
    y_pred = multi_layer_model_no_activation(X_torch)

    # Calculate the MSE loss between predicted values and actual values
    loss = loss_func(y_pred, y_torch)

    # Append the current loss to the loss history list
    loss_history.append(loss.item())

    # Zero the gradients to prevent accumulation
    multi_layer_optimizer_no_activation.zero_grad()

    # Backpropagate the gradients (compute gradients with respect to model parameters)
    loss.backward()

    # Update the model's parameters using the optimizer
    multi_layer_optimizer_no_activation.step()

# Plot the loss history over training epochs
plt.plot(loss_history)
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.show()

# Retrieve the final MSE loss value
mse = loss_history[-1]

# Make predictions on the input data using the trained model
y_hat = multi_layer_model_no_activation(X_torch).detach().numpy()

# Create a plot to visualize the original data, predicted line, and MSE value
plt.figure(figsize=(12, 7))
plt.title('PyTorch Model')
plt.scatter(X[:, 0], y, label='Data $(x1, y)$')
plt.plot(X[:, 0], y_hat.squeeze(), color='red', label='Predicted Line $y = f(x)$', linewidth=4.0)
plt.xlabel('$x$', fontsize=20)
plt.ylabel('$y$', fontsize=20)
plt.text(0, 0.70, 'MSE = {:.3f}'.format(mse), fontsize=20)
plt.grid(True)
plt.legend(fontsize=20)
plt.show()
# Define a custom neural network class named 'Net' with activation functions
class Net(nn.Module):
    def __init__(self, hidden_size):
        super(Net, self).__init__()
        self.hidden_size = hidden_size
        self.layer1 = nn.Linear(3, self.hidden_size)  # First linear layer
        self.layer2 = nn.Linear(self.hidden_size, 1)  # Second linear layer

    def forward(self, x):
        x = self.layer1(x)        # Forward pass through the first linear layer
        x = F.relu(x)             # Apply the ReLU activation function
        x = self.layer2(x)        # Forward pass through the second linear layer
        return x

# Create an instance of the 'Net' neural network with a hidden size of 100
multi_layer_model = Net(hidden_size=100)

# Set the learning rate for the optimizer
learning_rate = 1e-3

# Define the mean squared error (MSE) loss function for regression tasks
loss_func = torch.nn.MSELoss(reduction='mean')

# Initialize an Adam optimizer for the model's parameters
multi_layer_optimizer = torch.optim.Adam(multi_layer_model.parameters(), lr=learning_rate)

# Specify the number of training epochs
num_ep = 500

# Create an empty list to store loss values during training
loss_history = []

# Training loop
for ep in range(num_ep):
    # Forward pass: Predict the output (y_pred) using the current model
    y_pred = multi_layer_model(X_torch)

    # Calculate the MSE loss between predicted values and actual values
    loss = loss_func(y_pred, y_torch)

    # Append the current loss to the loss history list
    loss_history.append(loss.item())

    # Zero the gradients to prevent accumulation
    multi_layer_optimizer.zero_grad()

    # Backpropagate the gradients (compute gradients with respect to model parameters)
    loss.backward()

    # Update the model's parameters using the optimizer
    multi_layer_optimizer.step()

# Plot the loss history over training epochs
plt.plot(loss_history)
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.show()

# Retrieve the final MSE loss value
mse = loss_history[-1]

# Make predictions on the input data using the trained model
y_hat = multi_layer_model(X_torch).detach().numpy()

# Create a plot to visualize the original data, predicted line, and MSE value
plt.figure(figsize=(12, 7))
plt.title('PyTorch Model')
plt.scatter(X[:, 0], y, label='Data $(x1, y)$')
plt.plot(X[:, 0], y_hat.squeeze(), color='red', label='Predicted Line $y = f(x)$', linewidth=4.0)
plt.xlabel('$x$', fontsize=20)
plt.ylabel('$y$', fontsize=20)
plt.text(0, 0.70, 'MSE = {:.3f}'.format(mse), fontsize=20)
plt.grid(True)
plt.legend(fontsize=20)
plt.show()
