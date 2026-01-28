class SolverDFL:
    
    # abstract class for solvers
    def __init__(self):
        pass

    def solve(self, solver_inputs: dict):
        # abstract method to solve the optimization problem given the solver inputs
        raise NotImplementedError("This method should be overridden by subclasses")
    
    def evaluate_solution(self, solution, instance_inputs: dict) -> float:
        # abstract method to evaluate the solution on the true instance inputs
        raise NotImplementedError("This method should be overridden by subclasses")
    
    def save_solved_dataset(self, dataset_path: str, solved_dataset_path: str, verbose: bool = False):
        # abstract method to save a solved dataset
        raise NotImplementedError("This method should be overridden by subclasses")
