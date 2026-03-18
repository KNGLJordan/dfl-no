import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import copy
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from src.core.registry import MODELS

@MODELS.register("PFL_Baseline")
class PFL_Baseline(nn.Module):
    # A simple feedforward neural network for PFL baseline

    def __init__(self, input_dim, hidden_dim, output_dim, lr):

        super(PFL_Baseline, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        self.bn = nn.BatchNorm1d(hidden_dim)

        self.best_model_wts = copy.deepcopy(self.state_dict())
        self.best_val_loss = float('inf')
        self.train_loss_history = []
        self.val_loss_history = []

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)


    def forward(self, x):
        x = F.relu(self.bn(self.fc1(x)))
        x = self.fc3(x)
        return x
    
    def train_model(self, train_dataloader, val_dataloader, num_epochs, verbose = True):
   
        # Epoch loop
        if verbose:
            loop = tqdm(range(num_epochs), desc="Training")
        else:
            loop = range(num_epochs)
        
        for epoch in loop:

            # Train batch loop
            self.train()
            running_loss = 0.0

            for inputs, targets in train_dataloader:
                self.optimizer.zero_grad() # Zero the parameter gradients
                outputs = self(inputs) # Forward pass
                loss = self.criterion(outputs, targets) # Compute loss
                loss.backward() # Backward pass and optimization
                self.optimizer.step() # Update lr using optimizer
                running_loss += loss.item() * inputs.size(0) # Accumulate loss
            epoch_loss = running_loss / len(train_dataloader.dataset) # Avg loss per epoch
            self.train_loss_history.append(epoch_loss)
            
            # Val batch loop
            self.eval()
            val_running_loss = 0.0
            with torch.no_grad():
                for val_inputs, val_targets in val_dataloader:
                    val_outputs = self(val_inputs)
                    val_loss = self.criterion(val_outputs, val_targets)
                    val_running_loss += val_loss.item() * val_inputs.size(0)
            val_epoch_loss = val_running_loss / len(val_dataloader.dataset)
            self.val_loss_history.append(val_epoch_loss)

            # Saving best model weights
            if val_epoch_loss < self.best_val_loss:
                self.best_val_loss = val_epoch_loss
                self.best_model_wts = copy.deepcopy(self.state_dict())

            # Update progress bar
            if verbose:
                loop.set_postfix(train_loss=epoch_loss, val_loss=val_epoch_loss)
            
        # Load best model weights
        self.load_state_dict(self.best_model_wts)

    def get_loss_history(self):
        return self.train_loss_history, self.val_loss_history
    
    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        self.best_val_loss = float('inf')
        self.train_loss_history = []
        self.val_loss_history = []
        self.best_model_wts = copy.deepcopy(self.state_dict())

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def predict(self, X):
        self.eval()
        with torch.no_grad():
            outputs = self(X)
        return outputs
