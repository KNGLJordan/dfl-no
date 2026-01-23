import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from solvers.solver_KP import solve_KP
from torch.utils.data.dataset import Dataset

class PFLBaseline(nn.Module):
    # A simple feedforward neural network for PFL baseline

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PFLBaseline, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class Dataset_PFLBaseline(Dataset):
    # Dataset class for PFL Baseline model

    def __init__(self, X, values, weights, capacity, stochastic_target):
        self.X = X

        if stochastic_target == 'values':
            self.y = values
        elif stochastic_target == 'weights':
            self.y = weights
        elif stochastic_target == 'capacity':
            self.y = capacity.unsqueeze(1)  # Make capacity shape (N_instances, 1)
        else:
            raise ValueError("stochastic_target must be 'values', 'weights', or 'capacity'")
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
class DataLoader_PFLBaseline(DataLoader):
    # DataLoader for PFL Baseline model

    def __init__(self, dataset, batch_size=32, shuffle=True, num_workers=0):
        super(DataLoader_PFLBaseline, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )
    
def train_PFLBaseline(model, dataloader, criterion, optimizer, num_epochs, verbose = False):
    # Set training mode
    model.train()
    # Epoch loop
    for epoch in range(num_epochs):
        # Initialize loss for the epoch
        running_loss = 0.0
        # Batch loop
        for inputs, targets in dataloader:
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(inputs)
            # Compute loss
            loss = criterion(outputs, targets)
            # Backward pass and optimization
            loss.backward()
            # Update lr using optimizer
            optimizer.step()
            # Accumulate loss
            running_loss += loss.item() * inputs.size(0)
        # Compute average loss for the epoch
        epoch_loss = running_loss / len(dataloader.dataset)
        if verbose:
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

def evaluate_PFLBaseline(
        model, 
        X_test, # tensor of shape (N_instances, dim_x) containing the X features for each instance in the test set
        stochastic_target, # Options: 'values', 'weights', 'capacity'
        values, # tesnor of shape (N_instances, dim_values) containing the values for each instance in the test set
        weights, # tensor of shape (N_instances, dim_weights) containing the weights for each instance in the test set
        capacity, # tensor of shape (N_instances,) containing the capacity for each instance in the test set
        optimal_values # tensor of shape (N_instances,) containing the optimal solution (0/1) for each instance in the test set
    ):
    
    # Set evaluation mode
    model.eval()

    # Initialize predicted parameters
    pred_values = values
    pred_weights = weights
    pred_capacity = capacity

    # Disable gradient computation
    with torch.no_grad():
        # Forward pass
        outputs = model(X_test) # shape (N_instances, dim_output)

    if stochastic_target == 'values':
        pred_values = outputs
    elif stochastic_target == 'weights':
        pred_weights = outputs
    elif stochastic_target == 'capacity':
        # TODO : ensure capacity is positive
        pred_capacity = outputs.squeeze() # Squeeze to get shape (N_instances,)
    
    # Compute avg regret
    total_regret = 0.0
    for i in range(X_test.size(0)):
        # Solve KP with predicted parameters
        cost, _, _ = solve_KP(pred_values[i], pred_weights[i], pred_capacity[i])
        # Compute regret
        regret = abs(cost - optimal_values[i].item())
        total_regret += regret

    avg_regret = total_regret / X_test.size(0)

    return avg_regret