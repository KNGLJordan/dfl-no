import torch
import os 
import argparse

def generate_kp_data(
        n_samples=1000, 
        dim_x=5, 
        dim_y=50, 
        deg=5, 
        eps_bar=0.5,
        stochastic_target='values', # Options: 'values', 'weights', 'capacity'
        seed=0
    ):

    """
    Method for generating synthetic datasets for the Knapsack Problem (KP).

    Parameters:
    - n_samples: Number of samples to generate.
    - dim_x: Dimensionality of input features.
    - dim_y: Dimensionality of output targets.
    - deg: Degree of the polynomial relationship.
    - eps: Magnitude of multiplicative noise.
    - stochastic_target: Specifies which KP parameter is stochastic ('values', 'weights', or 'capacity').

    Returns:
    - X: Input feature matrix of shape (n_samples, dim_x).
    - values: Item values matrix of shape (n_samples, dim_y).
    - weights: Item weights matrix of shape (n_samples, dim_y).
    - capacity: Capacity vector of shape (n_samples,).
    

    Reference: Smart “Predict, then Optimize”; Adam N. Elmachtoub, Paul Grigas.
    """    

    # Set random seed for reproducibility
    torch.manual_seed(seed)

    # 0. Determine the dimensionality of the stochastic output
    # If target is capacity, we have a single value per instance
    actual_dim_target = 1 if stochastic_target == 'capacity' else dim_y

    # 1. Initialize random matrix B* (Bernoulli 0.5)
    B_star = torch.bernoulli(torch.full((actual_dim_target, dim_x), 0.5))    
    
    # 2. Input feature vectors generation x_i ~ N(0, I)
    X = torch.randn(n_samples, dim_x)
    
    # 3. Core stochastic mapping: Polynomial + Noise + Poisson
    # Equation: [(1/sqrt(p) * B*x + 3)^deg + 1]
    poly_term = ((X @ B_star.T) / torch.sqrt(torch.tensor(dim_x, dtype=torch.float)) + 3)**deg + 1
    noise = torch.empty(n_samples, actual_dim_target).uniform_(1 - eps_bar, 1 + eps_bar)
    y_lambda = poly_term * noise
    
    # Final stochastic values generated via Poisson distribution
    stochastic_signal = torch.poisson(y_lambda).float()

    # 4. Scenario assignment
    if stochastic_target == 'values':
        # Values depend on X
        values = stochastic_signal
        # Weights and Capacity are fixed
        weights = torch.randint(1, 15, (dim_y,)).float()
        weights = weights.repeat(n_samples, 1)
        capacity = torch.full((n_samples,), int(weights[0].sum().item() * 0.5))
        
    elif stochastic_target == 'weights':
        # Weights depend on X
        # Add 1 to avoid zero weights which would make the problem trivial # TODO: check if needed
        weights = stochastic_signal + 1
        #Values and Capacity are fixed
        values = torch.randint(10, 50, (dim_y,)).float()
        values = values.repeat(n_samples, 1)
        capacity = torch.full((n_samples,), int(weights.mean().item() * dim_y * 0.5))

    elif stochastic_target == 'capacity':
        # Capacity depends on X
        capacity = stochastic_signal.flatten()
        # Values and Weights are fixed
        values = torch.randint(10, 50, (dim_y,)).float()
        values = values.repeat(n_samples, 1)
        weights = torch.randint(1, 15, (dim_y,)).float()
        weights = weights.repeat(n_samples, 1)
        
    else:
        raise ValueError("stochastic_target must be 'values', 'weights', or 'capacity'")

    return X, values, weights, capacity

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Generate Knapsack Dataset")
    
    # Arguments for data generation
    parser.add_argument("--n_samples", type=int, default=1000, help="Number of samples")
    parser.add_argument("--dim_x", type=int, default=5, help="Dimension of input features")
    parser.add_argument("--dim_y", type=int, default=50, help="Number of items (dimension of output)")
    parser.add_argument("--deg", type=int, default=5, help="Degree of polynomial")
    parser.add_argument("--eps_bar", type=float, default=0.5, help="Noise magnitude")
    parser.add_argument("--target", type=str, default='values', choices=['values', 'weights', 'capacity'], help="Stochastic target")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Argument for save path
    parser.add_argument("--save_path", type=str, default="datasets/KP/knapsack_data.pt")
    
    args = parser.parse_args()

    print(f"Generating data:")
    print(f" - Number of samples: {args.n_samples}")
    print(f" - Input dimension (dim_x): {args.dim_x}")
    print(f" - Output dimension (dim_y): {args.dim_y}")
    print(f" - Polynomial degree: {args.deg}")
    print(f" - Noise magnitude (eps_bar): {args.eps_bar}")
    print(f" - Stochastic target: {args.target}")
    print(f" - Random seed: {args.seed}")
    
    # Generating KP insatnces
    X, values, weights, capacity = generate_kp_data(
        n_samples=args.n_samples,
        dim_x=args.dim_x,
        dim_y=args.dim_y,
        deg=args.deg,
        eps_bar=args.eps_bar,
        stochastic_target=args.target,
        seed=args.seed
    )

    # Creating save directory if it doesn't exist
    directory = os.path.dirname(args.save_path)
    if directory and not os.path.exists(directory):
        print(f"Creating directory: {directory}")
        os.makedirs(directory, exist_ok=True)

    # Saving the generated KP instances in a single .pt file
    torch.save({
        'X': X,
        'values': values,
        'weights': weights,
        'capacity': capacity
    }, args.save_path)
    
    print(f"Dataset saved at: {args.save_path}")

