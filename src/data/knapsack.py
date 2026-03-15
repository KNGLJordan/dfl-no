from src.data.dataset import DatasetDFL
import torch

from src.core.registry import DATASETS

class Knapsack_Dataset(DatasetDFL):
    
    # dataset class for knapsack problem instances
    def __init__(self, load_path):

        super().__init__(load_path)
           
        self.solver_inputs = {
            'values': self.data['values'],
            'weights': self.data['weights'],
            'capacity': self.data['capacity']
        }

    def get_true_solver_inputs(self, type='train'):

        if type == 'train':
            indices = self.train_indices
        elif type == 'val':
            indices = self.val_indices
        elif type == 'test':
            indices = self.test_indices
        else:
            raise ValueError("type must be 'train', 'val', or 'test'")

        return {
            'values': self.solver_inputs['values'][indices],
            'weights': self.solver_inputs['weights'][indices],
            'capacity': self.solver_inputs['capacity'][indices]
        }

@DATASETS.register("Knapsack_Values_Dataset")
class Knapsack_Values_Dataset(Knapsack_Dataset):

    def __init__(self, load_path):
        super().__init__(load_path)
        self.target = 'values'
    
    def get_solver_inputs_by_predictions(self, predictions, type='test'):

        if type == 'train':
            indices = self.train_indices
        elif type == 'val':
            indices = self.val_indices
        elif type == 'test':
            indices = self.test_indices
        else:
            raise ValueError("type must be 'train', 'val', or 'test'")

        return {
            'values': predictions,
            'weights': self.solver_inputs['weights'][indices],
            'capacity': self.solver_inputs['capacity'][indices]
        }

@DATASETS.register("Knapsack_Weights_Dataset")
class Knapsack_Weights_Dataset(Knapsack_Dataset):
    
    def __init__(self, load_path):
        super().__init__(load_path)
        self.target = 'weights'
    
    def get_solver_inputs_by_predictions(self, predictions, type='test'):

        if type == 'train':
            indices = self.train_indices
        elif type == 'val':
            indices = self.val_indices
        elif type == 'test':
            indices = self.test_indices
        else:
            raise ValueError("type must be 'train', 'val', or 'test'")

        return {
            'values': self.solver_inputs['values'][indices],
            'weights': predictions,
            'capacity': self.solver_inputs['capacity'][indices]
        }

@DATASETS.register("Knapsack_Capacity_Dataset")
class Knapsack_Capacity_Dataset(Knapsack_Dataset):
    
    def __init__(self, load_path):
        super().__init__(load_path)
        self.target = 'capacity'
    
    def get_solver_inputs_by_predictions(self, predictions, type='test'):

        if type == 'train':
            indices = self.train_indices
        elif type == 'val':
            indices = self.val_indices
        elif type == 'test':
            indices = self.test_indices
        else:
            raise ValueError("type must be 'train', 'val', or 'test'")

        return {
            'values': self.solver_inputs['values'][indices],
            'weights': self.solver_inputs['weights'][indices],
            'capacity': predictions.squeeze()
        }

       

