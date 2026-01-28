import torch.optim as optim
import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader
import copy
import argparse

from src.models.pfl_baseline import PFLBaseline
from src.datasets.dataset_KP import DatasetKPSolved
from src.solvers.solver_KP import Solver_KP
from src.metrics import compute_avg_regret
from src.utils import get_dataset, get_model, get_solver, map_prediction_to_solver_input

def parse_args():

    parser = argparse.ArgumentParser(description="Decision-Focused Learning Experiment Runner")

    # Data arguments
    parser.add_argument("--dataset", type=str, default="KP", choices=["KP"], help="Dataset name")
    parser.add_argument("--data_path", type=str, default="datasets/KP/knapsack_data_solved.pt", help="Path to .pt file")
    parser.add_argument("--target", type=str, default="values", choices=["values", "weights", "capacity"], help="Stochastic parameter to predict")
    
    # Model arguments
    parser.add_argument("--model", type=str, default="PFLBaseline", choices=["PFLBaseline"], help="Model architecture")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension size")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=2000, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    
    return parser.parse_args()

if __name__ == "__main__":

    # Set random seed for reproducibility
    args = parse_args()
    torch.manual_seed(args.seed)
    
    # Load dataset
    dataset = get_dataset(args.dataset, args.data_path, args.target)

    # Split train, val, test
    dataset.split(train_ratio=0.7, val_ratio=0.15)
    X_train, y_train = dataset.get_X('train'), dataset.get_y('train')
    X_val, y_val = dataset.get_X('val'), dataset.get_y('val')
    X_test = dataset.get_X('test')

    # Create dataloaders
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=args.batch_size, shuffle=False)

    # Initialize model
    input_dim = X_train.shape[1]
    model = get_model(args.model, input_dim, dataset, args.target)

    # Training hyperparameters
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Train model
    print(f"\nStarting training for {args.epochs} epochs...")
    model.train_model(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=args.epochs,
        verbose=True
    )

    # Predict-then-Optimize Evaluation
    print("\nStarting evaluation (Predict-then-Optimize)")

    # Get Ground Truth for testset
    true_solver_inputs = dataset.get_solver_inputs(type='test')
    true_optimal_values = dataset.get_optimal_solutions(type='test')

    # Optimization Solver 
    solver = get_solver(args.dataset)

    # Model prediction
    y_pred_test = model.predict(X_test)

    # Map predictions to solver inputs
    pred_solver_inputs = map_prediction_to_solver_input(
        true_solver_inputs, 
        y_pred_test, 
        args.dataset, 
        args.target
    )

    # Evaluate Regret on Test Set
    actual_obj_values = torch.zeros(X_test.shape[0])
    num_samples = X_test.shape[0]
    
    for i in range(num_samples):
        
        pred_input_i = {k: v[i] for k, v in pred_solver_inputs.items()}
        
        true_input_i = {k: v[i] for k, v in true_solver_inputs.items()}
        
        _, _, solution_items = solver.solve(pred_input_i)
        
        actual_val = solver.evaluate_solution(solution_items, true_input_i)
        
        actual_obj_values[i] = actual_val

    avg_regret = compute_avg_regret(actual_obj_values, true_optimal_values)
    
    print(f"RESULTS - {args.dataset} / {args.target}")
    print(f"Average Regret: {avg_regret:.4f}")



    



