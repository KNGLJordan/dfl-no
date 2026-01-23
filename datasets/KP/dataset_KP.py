import numpy as np
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
    np.random.seed(seed)

    # 0. Determine the dimensionality of the stochastic output
    # If target is capacity, we have a single value per instance
    actual_dim_target = 1 if stochastic_target == 'capacity' else dim_y

    # 1. Initialize random matrix B* (Bernoulli 0.5)
    B_star = np.random.binomial(n=1, p=0.5, size=(actual_dim_target, dim_x))
    
    # 2. Input feature vectors generation x_i ~ N(0, I)
    X = np.random.normal(0, 1, (n_samples, dim_x))
    
    # 3. Core stochastic mapping: Polynomial + Noise + Poisson
    # Equation: [(1/sqrt(p) * B*x + 3)^deg + 1]
    poly_term = ((X @ B_star.T) / np.sqrt(dim_x) + 3)**deg + 1
    noise = np.random.uniform(1 - eps_bar, 1 + eps_bar, (n_samples, actual_dim_target))
    y_lambda = poly_term * noise
    
    # Final stochastic values generated via Poisson distribution
    stochastic_signal = np.random.poisson(y_lambda).astype(float)

    # 4. Scenario assignment
    if stochastic_target == 'values':
        # Values depend on X, Weights and Capacity are fixed
        values = stochastic_signal
        weights = np.random.randint(1, 15, size=(dim_y,))
        # Broadcast fixed weights to all samples for consistency
        weights = np.tile(weights, (n_samples, 1))
        capacity = np.full(n_samples, int(np.sum(weights[0]) * 0.5))
        
    elif stochastic_target == 'weights':
        # Weights depend on X, Values and Capacity are fixed
        # Add 1 to avoid zero weights which would make the problem trivial
        weights = stochastic_signal + 1
        values = np.random.randint(10, 50, size=(dim_y,))
        values = np.tile(values, (n_samples, 1))
        # Capacity fixed based on the average of generated weights
        capacity = np.full(n_samples, int(np.mean(weights) * dim_y * 0.5))
        
    elif stochastic_target == 'capacity':
        # Capacity depends on X, Values and Weights are fixed
        capacity = stochastic_signal.flatten()
        values = np.random.randint(10, 50, size=(dim_y,))
        values = np.tile(values, (n_samples, 1))
        weights = np.random.randint(1, 15, size=(dim_y,))
        weights = np.tile(weights, (n_samples, 1))
        
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
    parser.add_argument("--save_path", type=str, default="datasets/KP/knapsack_data.npz", help="Path to save the dataset")

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

    # Saving the generated KP instances in a single .npz file
    np.savez(args.save_path, X=X, values=values, weights=weights, capacity=capacity)
    print(f"Dataset saved at: {args.save_path}")

