from ortools.linear_solver import pywraplp
import numpy as np
import time
import os
from solvers.solver_KP import solve_KP

def parse_instances(file_path):
    
    # check for file existence
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found")
    
    # Load the .npz file containing isntances of KP
    data = np.load(file_path) # NB: the file contains more then one instance
    
    # Extract features X, values, weights, and capacity
    X = data['X'] # dim : (N_instances, dim_x)
    values = data['values'] # dim : (N_instances, dim_values)
    weights = data['weights'] # dim : (N_instances, dim_weights)
    capacity = data['capacity'] # dim : (N_instances, )
    
    return X, values, weights, capacity

def solve_KP(values, weights, capacity):

    # Solver initialization
    solver = pywraplp.Solver.CreateSolver("SCIP")
    if not solver:
        print("Solver SCIP non found")
        return None, 0.0, []

    num_items = len(values)
    
    # 1. Variables: x[i] binary variable, 1 if the item i is taken, 0 otherwise
    x = [solver.BoolVar(f'x_{i}') for i in range(num_items)]
    
    # 2. Constraint: sum of weights of selected items <= capacity
    solver.Add(solver.Sum([weights[i] * x[i] for i in range(num_items)]) <= capacity)
    
    # 3. Obj function: maximize total value of selected items
    solver.Maximize(solver.Sum([values[i] * x[i] for i in range(num_items)]))
    
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
        return 0.0, exec_time, []

if __name__ == "__main__":

    dataset_path = "datasets/KP/knapsack_data.npz"
    
    try:

        # Load dataset parsing the file containing the instances of KP
        Xs, all_values, all_weights, all_capacities = parse_instances(dataset_path)

        # Select a particular instance to solve
        idx = 0 
        
        # Extract instance data
        values_i = all_values[idx]
        weights_i = all_weights[idx]
        capacity_i = all_capacities[idx]

        print(f"\nSolving instance {idx}:")
        print(f"- Number of items: {len(values_i)}")
        print(f"- Capacity: {capacity_i}")
        
        # Solve the selected instance
        best_val, runtime, sol = solve_KP(values_i, weights_i, capacity_i)
        
        print(f"Optimal value: {best_val}")
        print(f"Time required: {runtime:.4f} seconds")
        
    except FileNotFoundError as e:
        print(e)
        print("Dataset file not found")
    except Exception as e:
        print(f"An error during loading or solving the instance occured: {e}")