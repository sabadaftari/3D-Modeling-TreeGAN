import torch.nn as nn
import torch
# Define the Discriminator
class TreeGANDiscriminator(nn.Module):
    """
    TreeGAN Discriminator for distinguishing between real and generated 3D point clouds.
    
    Args:
        features (int): Number of features of the input (e.g., 3 for 3D point coordinates).
    
    Methods:
        forward(x: Tensor) -> Tensor: Passes the input point cloud `x` through the discriminator 
                                      and returns a real/fake score.
    """
    def __init__(self, features: int) -> None:
        """
        Initializes the TreeGAN Discriminator model.
        
        Args:
            features (int): Number of features of the input point cloud.
        """
        super(TreeGANDiscriminator, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(features, 64),          # Input layer to 64 units
            nn.LeakyReLU(0.2),                # LeakyReLU activation with negative slope 0.2
            nn.Linear(64, 128),               # 64 units to 128 units
            nn.LeakyReLU(0.2),                # Another LeakyReLU activation
            nn.Linear(128, 1)                 # Final layer with 1 output for real/fake score
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the discriminator network.
        
        Args:
            x (torch.Tensor): Input point cloud of shape [batch_size, num_points, features].
            
        Returns:
            torch.Tensor: A score for each input point cloud. The output has shape [batch_size].
        """
        return self.features(x).squeeze()  # Removes dimensions of size 1, returning [batch_size]