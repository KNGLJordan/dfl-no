from src.datasets.dataset import DatasetDFL, DatasetSolvedDFL
import torch

class DatasetKP(DatasetDFL):
    
    # dataset class for knapsack problem instances
    def __init__(self, load_path, stochastic_target='values'):

        super().__init__(load_path)

        self.num_samples = self.data['X'].shape[0]

        self.X = self.data['X']

        if stochastic_target not in ['values', 'weights', 'capacity']:
            raise ValueError("stochastic_target must be 'values', 'weights', or 'capacity'")
        else:
            if stochastic_target == 'values':
                self.y = self.data['values']
            elif stochastic_target == 'weights':
                self.y = self.data['weights']
            elif stochastic_target == 'capacity':
                self.y = self.data['capacity']
    
        self.solver_inputs = {
            'values': self.data['values'],
            'weights': self.data['weights'],
            'capacity': self.data['capacity']
        }

    def split(self, train_ratio=0.8, val_ratio=0.1):
        n_train = int(self.num_samples * train_ratio)
        n_val = int(self.num_samples * val_ratio)
        n_test = self.num_samples - n_train - n_val

        indices = torch.randperm(self.num_samples)

        self.train_indices = indices[:n_train]
        self.val_indices = indices[n_train:n_train + n_val]
        self.test_indices = indices[n_train + n_val:]

    def get_X(self, type='train'):
        if type == 'train':
            return self.X[self.train_indices]
        elif type == 'val':
            return self.X[self.val_indices]
        elif type == 'test':
            return self.X[self.test_indices]
        else:
            raise ValueError("type must be 'train', 'val', or 'test'")

    def get_y(self, type='train'):
        if type == 'train':
            return self.y[self.train_indices]
        elif type == 'val':
            return self.y[self.val_indices]
        elif type == 'test':
            return self.y[self.test_indices]
        else:
            raise ValueError("type must be 'train', 'val', or 'test'")

    def get_solver_inputs(self, type='train'):

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
    
class DatasetKPSolved(DatasetKP):
    
    # dataset class for knapsack problem instances with optimal solutions
    def __init__(self, load_path, stochastic_target='values'):

        super().__init__(load_path, stochastic_target)

        self.optimal_solutions = self.data['optimal_values']

    def get_optimal_solutions(self, type='train'):
        if type == 'train':
            return self.optimal_solutions[self.train_indices]
        elif type == 'val':
            return self.optimal_solutions[self.val_indices]
        elif type == 'test':
            return self.optimal_solutions[self.test_indices]
        else:
            raise ValueError("type must be 'train', 'val', or 'test'")
        

# Test main 

if __name__ == "__main__":
    dataset_path = "datasets/KP/knapsack_data_solved.pt"
    dataset = DatasetKPSolved(load_path=dataset_path, stochastic_target='values')
    dataset.split(train_ratio=0.7, val_ratio=0.15)

    X_train = dataset.get_X(type='train')
    y_train = dataset.get_y(type='train')
    solver_inputs_train = dataset.get_solver_inputs(type='train')
    optimal_solutions_train = dataset.get_optimal_solutions(type='train')

    print(f"Train X shape: {X_train.shape}")
    print(f"Train y shape: {y_train.shape}")
    print(f"Train solver inputs: values shape {solver_inputs_train['values'].shape}, weights shape {solver_inputs_train['weights'].shape}, capacity shape {solver_inputs_train['capacity'].shape}")
    print(f"Train optimal solutions shape: {optimal_solutions_train.shape}")
