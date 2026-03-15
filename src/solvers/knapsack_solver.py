from ortools.linear_solver import pywraplp
import torch
import time
import os
from src.solvers.solver import SolverDFL
from src.core.registry import SOLVERS

@SOLVERS.register("Knapsack_Solver")
class Knapsack_Solver(SolverDFL):

    def __init__(self):
        super().__init__()

    def solve(self, solver_inputs: dict):

        # Increment solver call count
        self.solver_calls += 1

        # Convert input Tensors to lists
        val_list = solver_inputs["values"].tolist() 
        weight_list = solver_inputs["weights"].tolist()
        cap_val = solver_inputs["capacity"].item() 

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
        
    def evaluate_solution(self, solution, instance_inputs: dict) -> float:
       
        # 1. Estrazione dei dati reali
        # Assumiamo che siano tensori o scalari, li convertiamo in float/list per sicurezza
        true_values = instance_inputs['values']
        true_weights = instance_inputs['weights']
        true_capacity = instance_inputs['capacity']
        
        # Se sono tensori PyTorch, convertiamoli in liste/float per calcoli rapidi
        if isinstance(true_values, torch.Tensor):
            true_values = true_values.tolist()
        if isinstance(true_weights, torch.Tensor):
            true_weights = true_weights.tolist()
        if isinstance(true_capacity, torch.Tensor):
            true_capacity = true_capacity.item()

        # 2. Calcolo del peso totale reale della soluzione
        current_weight = sum([w * x for w, x in zip(true_weights, solution)])
        
        # 3. Controllo Ammissibilità (Feasibility Check)
        # Se la soluzione viola la capacità REALE, è invalida.
        if current_weight > true_capacity:
            return 0.0 # O un valore negativo se vuoi penalizzare fortemente
        
        # 4. Calcolo del Valore Reale (Objective Function)
        actual_obj_value = sum([v * x for v, x in zip(true_values, solution)])
        
        return actual_obj_value

    