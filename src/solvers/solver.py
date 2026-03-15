class SolverDFL:
    
    # abstract class for solvers
    def __init__(self):
        self.solver_calls = 0

    def get_solver_calls(self):
        return self.solver_calls

    def solve(self, solver_inputs: dict):
        # abstract method to solve the optimization problem given the solver inputs
        raise NotImplementedError("This method should be overridden by subclasses")
    
    def evaluate_solution(self, solution, instance_inputs: dict) -> float:
        # abstract method to evaluate the solution on the true instance inputs
        raise NotImplementedError("This method should be overridden by subclasses")
    

