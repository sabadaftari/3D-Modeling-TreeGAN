import os
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader

def Point_DataLoader(batch_size: int, dataset_path: str = "src/data/ModelNet10") -> tuple[DataLoader, DataLoader]:
    """
    Loads the ModelNet10 dataset and returns training and validation dataloaders.
    
    Args:
        batch_size (int): The number of samples to load per batch.
        dataset_path (str): Path to the directory containing the dataset.
        
    Returns:
        tuple[DataLoader, DataLoader]: A tuple containing:
            - train_loader (DataLoader): A DataLoader for training data.
            - val_loader (DataLoader): A DataLoader for validation data.
    """
    # Ensure the dataset directory exists
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"The dataset path '{dataset_path}' does not exist.")
    
    # Load the training and validation datasets
    train_dataset = ModelNet(root=dataset_path, name='10', train=True)
    validation_dataset = ModelNet(root=dataset_path, name='10', train=False)

    # Create DataLoader for training and validation datasets
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    
    return train_dataloader, val_dataloader