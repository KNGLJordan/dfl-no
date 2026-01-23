from solvers.solver_KP import parse_instances, solve_KP, parse_solved_instances
from src.pfl_baseline import PFLBaseline, Dataset_PFLBaseline, DataLoader_PFLBaseline, train_PFLBaseline, evaluate_PFLBaseline
import torch.optim as optim
import torch.nn as nn
import torch

def split_np_array(arr, train_size, val_size):
    train_arr = arr[:train_size]
    val_arr = arr[train_size:train_size + val_size]
    test_arr = arr[train_size + val_size:]
    return train_arr, val_arr, test_arr


if __name__ == "__main__":

    # dataset path
    dataset_path = "datasets/KP/knapsack_data_solved.npz"

    # parse dataset
    print("Loading dataset...")
    X, values, weights, capacity, optimal_values, solve_times = parse_solved_instances(dataset_path)
    num_samples = len(values)

    # convert to torch tensors
    X = torch.tensor(X, dtype=torch.float32)
    values = torch.tensor(values, dtype=torch.float32)
    weights = torch.tensor(weights, dtype=torch.float32)
    capacity = torch.tensor(capacity, dtype=torch.float32)
    optimal_values = torch.tensor(optimal_values, dtype=torch.float32)
    solve_times = torch.tensor(solve_times, dtype=torch.float32)

    # Split train, val, test
    train_size = int(0.7 * num_samples)
    val_size = int(0.15 * num_samples)
    test_size = num_samples - train_size - val_size

    X_train, X_val, X_test = split_np_array(X, train_size, val_size)
    values_train, values_val, values_test = split_np_array(values, train_size, val_size)
    weights_train, weights_val, weights_test = split_np_array(weights, train_size, val_size)
    capacity_train, capacity_val, capacity_test = split_np_array(capacity, train_size, val_size)
    optimal_values_train, optimal_values_val, optimal_values_test = split_np_array(optimal_values, train_size, val_size)
    solve_times_train, solve_times_val, solve_times_test = split_np_array(solve_times, train_size, val_size)

    # Select stochastic target (between 'values', 'weights', 'capacity')
    stochastic_target = 'values'

    # Create dataset
    dataset_train = Dataset_PFLBaseline(X_train, values_train, weights_train, capacity_train, stochastic_target)

    # Create dataloader
    dataloader_train = DataLoader_PFLBaseline(dataset_train, batch_size=32, shuffle=True)

    # Initialize model
    input_dim = X.shape[1]

    hidden_dim = 64

    if stochastic_target == 'capacity':
        output_dim = 1
    elif stochastic_target == 'values':
        output_dim = values.shape[1]
    elif stochastic_target == 'weights':
        output_dim = weights.shape[1]

    model = PFLBaseline(input_dim, hidden_dim, output_dim)

    # Training hyperparameters
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 2000

    # Train model
    print(f"Training model for {num_epochs} epochs...")
    train_PFLBaseline(model, dataloader_train, criterion, optimizer, num_epochs)

    # Evaluate on test set
    print("Evaluating model on test set...")
    avg_regret = evaluate_PFLBaseline(
        model = model,
        X_test = X_test,
        stochastic_target=stochastic_target,
        values=values_test,
        weights=weights_test,
        capacity=capacity_test,
        optimal_values=optimal_values_test
    )

    # Print results
    print(f"Average regret on test set: {avg_regret:.4f}")



    



