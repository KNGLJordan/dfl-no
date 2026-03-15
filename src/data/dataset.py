import torch

class DatasetDFL:

    # abstarct class for datasets used in DFL experiments
    def __init__(self, load_path):

        self.load_path = load_path
        self.data = torch.load(load_path)

        self.num_samples = self.data['X'].shape[0]

        self.X = self.data['X']
        self.y = self.data['y']

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
    
    def get_true_solver_inputs(self, type='train'):
        raise NotImplementedError

    def get_solver_inputs_by_predictions(self, predictions):
        raise NotImplementedError



