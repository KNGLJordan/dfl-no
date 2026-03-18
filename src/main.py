import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader
import copy
import argparse

from src.core.registry import DATASETS, MODELS, SOLVERS

import src.data.knapsack
import src.models.pfl_baseline
import src.solvers.knapsack_solver

from src.utils.transforms import TorchStandardScaler

from src.utils.metrics import compute_avg_regret

import wandb

def parse_args():

    parser = argparse.ArgumentParser(description="Decision-Focused Learning Experiment Runner")

    # Data arguments
    parser.add_argument("--dataset", type=str, default="Knapsack_Values_Dataset", choices=["Knapsack_Values_Dataset", "Knapsack_Weights_Dataset", "Knapsack_Capacity_Dataset"], help="Dataset class name")
    parser.add_argument("--data_path", type=str, default="datasets/KP/knapsack_values_data.pt", help="Path to .pt file")
    
    # Model arguments
    parser.add_argument("--model", type=str, default="PFL_Baseline", choices=["PFL_Baseline"], help="Model architecture class name")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension size")
    
    # Training arguments
    parser.add_argument("--train_split", type=float, default=0.7, help="Proportion of data for training")
    parser.add_argument("--val_split", type=float, default=0.15, help="Proportion of data for validation")
    parser.add_argument("--epochs", type=int, default=2000, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--normalize_input", type=int, default=0, choices=[0,1], help="Whether to normalize input features")
    parser.add_argument("--normalize_output", type=int, default=0, choices=[0,1], help="Whether to normalize output targets")

    # Solver arguments
    parser.add_argument("--solver", type=str, default="Knapsack_Solver", choices=["Knapsack_Solver"], help="Optimization solver class name")
    
    return parser.parse_args()

if __name__ == "__main__":

    # Initialize Weights & Biases
    wandb.init(entity="giordani-francesco2002-university-of-bologna", project="dfl-no")

    # Set random seed for reproducibility
    args = parse_args()
    torch.manual_seed(args.seed)
    
    # Load dataset
    DatasetClass = DATASETS.get(args.dataset)
    dataset = DatasetClass(args.data_path)

    # Split train, val, test
    dataset.split(train_ratio=args.train_split, val_ratio=args.val_split)
    X_train, y_train = dataset.get_X('train'), dataset.get_y('train')
    X_val, y_val = dataset.get_X('val'), dataset.get_y('val')
    X_test = dataset.get_X('test')

    # Normalize input data
    if args.normalize_input:
        scaler_X = TorchStandardScaler()
        X_train = scaler_X.fit_transform(X_train)
        X_val = scaler_X.transform(X_val)
        X_test = scaler_X.transform(X_test)

    # Normalize output data
    if args.normalize_output:
        scaler_y = TorchStandardScaler()
        y_train = scaler_y.fit_transform(y_train)
        y_val = scaler_y.transform(y_val)
       
    # Create dataloaders
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=args.batch_size, shuffle=False)

    # Initialize model
    input_dim = X_train.shape[1]
    ModelClass = MODELS.get(args.model)
    if args.model == "PFL_Baseline":
        model = ModelClass(input_dim=input_dim, hidden_dim=args.hidden_dim, output_dim=(y_train.shape[1] if len(y_train.shape) > 1 else 1), lr=args.lr)
    else:
        print(f"To think about how to handle this part of the main from a software engineering PoV, since the model class may have different init args...")
        exit()

    # Train model
    print(f"\nStarting training for {args.epochs} epochs...")
    model.train_model(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        num_epochs=args.epochs,
        verbose=True
    )

    # Predict-then-Optimize Evaluation
    print("\nStarting evaluation (Predict-then-Optimize)")

    # Get Ground Truth for testset
    true_solver_inputs = dataset.get_true_solver_inputs(type='test')

    # Model prediction and denormalization
    y_pred_test = model.predict(X_test)
    if args.normalize_output:
        y_pred_test = scaler_y.inverse_transform(y_pred_test)
    # Map predictions to solver inputs (i.e y_hat)
    pred_solver_inputs = dataset.get_solver_inputs_by_predictions(y_pred_test, type='test')

    # Optimization Solver 
    SolverClass = SOLVERS.get(args.solver)
    solver = SolverClass()

    # Evaluate Regret on Test Set
    true_optimal_values = torch.zeros(X_test.shape[0])
    actual_obj_values = torch.zeros(X_test.shape[0])
    num_samples = X_test.shape[0]
    
    for i in range(num_samples):
        
        # Compute the cost under true parameters of the optimal solution under true parameters (i.e. g(y_i,z*(y_i)))
        true_input_i = {k: v[i] for k, v in true_solver_inputs.items()}
        optimal_cost, _, _ = solver.solve(true_input_i)
        true_optimal_values[i] = optimal_cost
        
        # Compute the cost under true parameters of the optimal solution under estimated parameters (i.e. g(y_i,z*(y_hat_i)))
        pred_input_i = {k: v[i] for k, v in pred_solver_inputs.items()}
        _, _, decision_by_prediction = solver.solve(pred_input_i)
        actual_cost = solver.evaluate_solution(decision_by_prediction, true_input_i)
        actual_obj_values[i] = actual_cost

    avg_regret = compute_avg_regret(actual_obj_values, true_optimal_values)
    
    print(f"RESULTS - {args.dataset}")
    print(f"Average Regret: {avg_regret:.4f}")

    # Log results to Weights & Biases
    if wandb.run is not None:
        wandb.log({"Average_Regret": avg_regret})



    



