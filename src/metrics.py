import torch 

def compute_avg_regret(predicted_solutions, optimal_solutions):
    
    # Ensure inputs are tensors
    if not isinstance(predicted_solutions, torch.Tensor):
        predicted_solutions = torch.tensor(predicted_solutions)
    if not isinstance(optimal_solutions, torch.Tensor):
        optimal_solutions = torch.tensor(optimal_solutions)

    # Compute regret for each instance
    regrets = torch.abs(optimal_solutions - predicted_solutions)

    # Compute average regret
    avg_regret = torch.mean(regrets).item()

    return avg_regret
   