import torch
import torch.nn as nn
from src.models import TreeGCN

# Define the Generator
class TreeGANGenerator(nn.Module):
    """
    TreeGAN Generator for generating 3D point clouds from noise vectors using TreeGCN layers.
    
    Args:
        features (list): List of integers representing the feature sizes for each TreeGCN layer.
        degrees (list): List of integers representing the degree (number of branches) for each TreeGCN layer.
        batch_size (int): The number of samples per batch.
    
    Methods:
        forward(z: Tensor) -> Tensor: Passes the noise vector `z` through the GCN layers to generate 
                                      the 3D point cloud.
    """
    def __init__(self, features: list[int], degrees: list[int], batch_size: int) -> None:
        """
        Initializes the TreeGAN Generator model with a series of TreeGCN layers and a final fully connected layer.

        Args:
            features (list): Feature dimensions for each TreeGCN layer.
            degrees (list): Degrees (number of branches) for each TreeGCN layer.
            batch_size (int): Batch size of samples.
        """
        super(TreeGANGenerator, self).__init__()
        self.batch_size = batch_size
        self.layer_num = len(features) - 1
        self.vertex_num = 1  # Starting with 1 vertex (root)
        self.gcn = nn.Sequential()

        # Create a TreeGCN layer for each feature set
        for inx in range(self.layer_num):
            self.gcn.add_module(f'TreeGCN_{inx}', TreeGCN(features, degrees, inx))
            self.vertex_num *= degrees[inx]  # Update vertex count for the next layer

        # Final fully connected layer to map features to 3D coordinates (x, y, z)
        self.fc_final = nn.Linear(features[-1], 3)  # Last feature size to 3 for point coordinates

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for generating 3D point clouds.
        
        Args:
            z (torch.Tensor): Noise vector of shape [batch_size, latent_dim].
        
        Returns:
            torch.Tensor: Generated 3D point cloud with shape [total_points, 3].
        """
        feat = self.gcn(z)  # Pass through GCN layers
        final_output = self.fc_final(feat)  # Map to 3D coordinates
        return final_output.view(-1, 3)  # Reshape to [total_points, 3] (x, y, z)



