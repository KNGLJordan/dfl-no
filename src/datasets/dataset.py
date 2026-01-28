import torch

class DatasetDFL:

    # abstarct class for datasets used in DFL experiments
    def __init__(self, load_path):
        self.load_path = load_path
        self.data = torch.load(load_path)

    def split(self, train_ratio=0.8, val_ratio=0.1):
        raise NotImplementedError

    def get_X(self, type='train'):
        raise NotImplementedError
    
    def get_y(self, type='train'):
        raise NotImplementedError
    
    def get_solver_inputs(self, type='train'):
        raise NotImplementedError
    
class DatasetSolvedDFL(DatasetDFL):

    # abstract class for datasets used in DFL experiments with solved instances
    def __init__(self, load_path):
        super().__init__(load_path)

    def get_optimal_solutions(self, type='train'):
        raise NotImplementedError


