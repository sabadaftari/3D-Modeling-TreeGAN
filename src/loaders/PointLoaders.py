from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader

# Dataset loading
def Point_DataLoader(batch_size, dataset_path="src/data/ModelNet10"):
    """
    Loads the ModelNet10 dataset and returns a DataLoader.
    
    Args:
        batch_size (int): Batch size for training.
        dataset_path (str): Path to the dataset directory.
        
    Returns:
        dataloader (DataLoader): A DataLoader for the ModelNet dataset.
    """
    dataset = ModelNet(root=dataset_path, name='10', train=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
