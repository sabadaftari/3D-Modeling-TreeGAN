from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader

# Dataset loading
def Point_DataLoader(batch_size, dataset_path="src/data/ModelNet10"):
    """
    Loads the ModelNet10 dataset and returns training dataloader and validation dataloader.
    
    Args:
        batch_size (int): Batch size for training.
        dataset_path (str): Path to the dataset directory.
        
    Returns:
        train_loader (DataLoader): A DataLoader for training data.
        val_loader (DataLoader): A DataLoader for validation data.
    """
    train_dataset = ModelNet(root=dataset_path, name='10', train=True)
    validation_dataset = ModelNet(root=dataset_path, name='10', train=False)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
    
    return train_dataloader, val_dataloader
