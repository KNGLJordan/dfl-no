import torch

def split_tensor(tensor: torch.Tensor, train_size: int, val_size: int):

    if not isinstance(tensor, torch.Tensor):
        raise ValueError("Input must be a torch Tensor")
    
    if train_size + val_size > len(tensor):
        raise ValueError("Train size and validation size exceed total tensor length")

    train = tensor[:train_size]
    val = tensor[train_size:train_size + val_size]
    test = tensor[train_size + val_size:]
    
    return train, val, test