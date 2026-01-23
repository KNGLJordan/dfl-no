from ortools.linear_solver import pywraplp
import torch
import time
import os

def parse_instances(file_path):
    
    # check for file existence
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found")
    
    # Load the .npz file containing isntances of KP
    data = torch.load(file_path) # NB: the file contains more then one instance
    
    # Extract features X, values, weights, and capacity
    X = data['X'] # dim : (N_instances, dim_x)
    values = data['values'] # dim : (N_instances, dim_values)
    weights = data['weights'] # dim : (N_instances, dim_weights)
    capacity = data['capacity'] # dim : (N_instances, )
    
    return X, values, weights, capacity

def parse_solved_instances(file_path):
    
    # check for file existence
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found")
    
    # Load the .npz file containing isntances of KP
    data = torch.load(file_path) # NB: the file contains more then one instance
    
    # Extract features X, values, weights, capacity, optimal_values, solve_times
    X = data['X'] # dim : (N_instances, dim_x)
    values = data['values'] # dim : (N_instances, dim_values)
    weights = data['weights'] # dim : (N_instances, dim_weights)
    capacity = data['capacity'] # dim : (N_instances, )
    optimal_values = data['optimal_values'] # dim : (N_instances, )
    solve_times = data['solve_times'] # dim : (N_instances, )
    
    return X, values, weights, capacity, optimal_values, solve_times

def solve_KP(values:torch.Tensor, weights: torch.Tensor, capacity: torch.Tensor):

    # Convert input Tensors to lists
    val_list = values.tolist() 
    weight_list = weights.tolist()
    cap_val = capacity.item() 

    # Solver initialization
    solver = pywraplp.Solver.CreateSolver("SCIP")
    if not solver:
        print("Solver SCIP non found")
        return None, 0.0, []

    # Instance size
    num_items = len(val_list)
    
    # 1. Variables: x[i] binary variable, 1 if the item i is taken, 0 otherwise
    x = [solver.BoolVar(f'x_{i}') for i in range(num_items)]
    
    # 2. Constraint: sum of weights of selected items <= capacity
    solver.Add(solver.Sum([weight_list[i] * x[i] for i in range(num_items)]) <= cap_val)
    
    # 3. Obj function: maximize total value of selected items
    solver.Maximize(solver.Sum([val_list[i] * x[i] for i in range(num_items)]))
    
    # Solving and computing execution time
    start_time = time.time()
    status = solver.Solve()
    end_time = time.time()
    
    exec_time = end_time - start_time
    
    # Result
    if status == pywraplp.Solver.OPTIMAL:
        # Optimal solution found
        obj_value = solver.Objective().Value()
        # Selected items
        solution_items = [int(x[i].solution_value()) for i in range(num_items)]
        return obj_value, exec_time, solution_items
    else:
        print("No optimal solution has been found")
        return float('inf'), exec_time, []
    
def save_solved_dataset_KP(dataset_path, solved_dataset_path, verbose=False):

    try:

        times = []
        optimal_values = []

        # Load dataset parsing the file containing the instances of KP
        Xs, all_values, all_weights, all_capacities = parse_instances(dataset_path)

        # Instances loop
        for idx in range(len(all_values)):
        
            # Extract instance data
            values_i = all_values[idx]
            weights_i = all_weights[idx]
            capacity_i = all_capacities[idx]
            
            # Solve the selected instance
            best_val, runtime, sol = solve_KP(values_i, weights_i, capacity_i)

            if verbose:
                print(f"\nSolved instance {idx}: optimal value = {best_val}, time = {runtime:.4f} seconds")

            optimal_values.append(best_val)
            times.append(runtime)

        # Transform lists to torch Tenosrs
        optimal_values_tensor = torch.tensor(optimal_values, dtype=torch.float32)
        times_tensor = torch.tensor(times, dtype=torch.float32)

        # Save the results in a new .pt file
        torch.save({
            'X': Xs,
            'values': all_values,
            'weights': all_weights,
            'capacity': all_capacities,
            'optimal_values': optimal_values_tensor,
            'solve_times': times_tensor
        }, solved_dataset_path)
        
    except FileNotFoundError as e:
        print(e)
        print("Dataset file not found")
    except Exception as e:
        print(f"An error during loading or solving the instance occured: {e}")

if __name__ == "__main__":

    dataset_path = "datasets/KP/knapsack_data.pt"

    solved_dataset_path = "datasets/KP/knapsack_data_solved.pt"

    save_solved_dataset_KP(dataset_path, solved_dataset_path, verbose=True)
    