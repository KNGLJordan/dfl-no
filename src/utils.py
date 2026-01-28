from src.datasets.dataset_KP import DatasetKPSolved
from src.solvers.solver_KP import Solver_KP
from src.models.pfl_baseline import PFLBaseline

def get_dataset(name, path, target):

    print(f"Loading dataset: {name} from {path} with target {target}...")
    
    if name == "KP":
        return DatasetKPSolved(load_path=path, stochastic_target=target)
    else:
        raise ValueError(f"Unknown dataset type: {name}")

def get_solver(name):
    if name == "KP":
        return Solver_KP()
    else:
        raise ValueError(f"Unknown solver type: {name}")

def get_model(name, input_dim, dataset, target):

    
    output_dim = 0
    if isinstance(dataset, DatasetKPSolved): # Logica specifica per KP
        if target == 'capacity':
            output_dim = 1
        elif target == 'values':
            output_dim = dataset.solver_inputs['values'].shape[1]
        elif target == 'weights':
            output_dim = dataset.solver_inputs['weights'].shape[1]
    
    print(f"Initializing {name}: Input dim {input_dim} -> Output dim {output_dim}")

    if name == "PFLBaseline":
        hidden_dim = 128 
        return PFLBaseline(input_dim, hidden_dim, output_dim)
    else:
        raise ValueError(f"Unknown model type: {name}")

def map_prediction_to_solver_input(inputs, prediction, dataset_type, target):

    modified_inputs = inputs.copy()
    
    if dataset_type == "KP":
        if target == 'values':
            modified_inputs['values'] = prediction
        elif target == 'weights':
            modified_inputs['weights'] = prediction
        elif target == 'capacity':
            modified_inputs['capacity'] = prediction.squeeze()
            
    return modified_inputs