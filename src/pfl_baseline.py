import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import copy

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
    
def train_PFLBaseline(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs, verbose = True):
   
    # History 
    train_loss_history = []
    val_loss_history = []

    # Best results
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    
    # Epoch loop
    if verbose:
        loop = tqdm(range(num_epochs), desc="Training")
    else:
        loop = range(num_epochs)
    
    for epoch in loop:

        # Train batch loop
        model.train()
        running_loss = 0.0

        for inputs, targets in train_dataloader:
            optimizer.zero_grad() # Zero the parameter gradients
            outputs = model(inputs) # Forward pass
            loss = criterion(outputs, targets) # Compute loss
            loss.backward() # Backward pass and optimization
            optimizer.step() # Update lr using optimizer
            running_loss += loss.item() * inputs.size(0) # Accumulate loss
        epoch_loss = running_loss / len(train_dataloader.dataset) # Avg loss per epoch
        train_loss_history.append(epoch_loss)
        
        # Val batch loop
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for val_inputs, val_targets in val_dataloader:
                val_outputs = model(val_inputs)
                val_loss = criterion(val_outputs, val_targets)
                val_running_loss += val_loss.item() * val_inputs.size(0)
        val_epoch_loss = val_running_loss / len(val_dataloader.dataset)
        val_loss_history.append(val_epoch_loss)

        # Saving best model weights
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())

        # Update progress bar
        if verbose:
            loop.set_postfix(train_loss=epoch_loss, val_loss=val_epoch_loss)
        
    # Load best model weights
    model.load_state_dict(best_model_wts)

    # Return training history
    return train_loss_history, val_loss_history


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