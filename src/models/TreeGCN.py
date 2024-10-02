import torch
import torch.nn as nn

class TreeGCN(nn.Module):
    """
    Tree-based Graph Convolutional Network (TreeGCN) layer for generating 3D point clouds.
    
    Args:
        features (list[int]): A list containing the number of features for each layer of the generator.
        degrees (list[int]): A list containing the degree (number of child nodes) for each layer in the tree structure.
        layer_idx (int): The current layer index.
    
    Methods:
        forward(z: Tensor) -> Tensor: Processes the input `z` through root and loop transformations, 
                                       applying 1D convolutions and LeakyReLU activation.
    """
    
    def __init__(self, features: list[int], degrees: list[int], layer_idx: int) -> None:
        """
        Initializes the TreeGCN layer with root transformations, loop transformations, and convolutional layers.

        Args:
            features (list[int]): Feature dimensions for each layer.
            degrees (list[int]): Degrees (number of child nodes) for each layer.
            layer_idx (int): Index of the current layer in the TreeGCN.
        """
        super(TreeGCN, self).__init__()
        self.layer_idx = layer_idx
        
        # Root node transformation
        self.fc_root = nn.Linear(features[layer_idx], features[layer_idx + 1], bias=False)
        
        # 1D convolution layers for enhancing local feature learning
        self.conv1 = nn.Conv1d(in_channels=features[layer_idx + 1], 
                               out_channels=features[layer_idx + 1], 
                               kernel_size=3, 
                               padding=1)  # Ensuring output has the same size
        
        self.conv2 = nn.Conv1d(in_channels=features[layer_idx + 1], 
                               out_channels=features[layer_idx + 1], 
                               kernel_size=3, 
                               padding=1)
        
        # Loop node transformation
        self.fc_loop = nn.Sequential(
            nn.Linear(features[layer_idx + 1], degrees[layer_idx] * features[layer_idx + 1], bias=False),
            nn.LeakyReLU(0.2),
            nn.Linear(degrees[layer_idx] * features[layer_idx + 1], features[layer_idx + 1], bias=False)
        )
        
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """ 
        Applies the TreeGCN transformation to the input `z`, performing both root and loop transformations, 
        and Conv1d layers for improved local feature extraction.
        
        Args:
            z (torch.Tensor): Input tensor of shape [batch_size, vertex_num, features].
        
        Returns:
            torch.Tensor: Output tensor after applying the transformations and activations.
        """
        # Root transformation
        root = self.fc_root(z)
        
        # Reshape the output to fit Conv1d input
        root = root.permute(0, 2, 1)
        
        # Apply the first Conv1d layer
        conv_out1 = self.conv1(root)
        
        # Apply the second Conv1d layer
        conv_out2 = self.conv2(conv_out1)
        
        # Permute back to the original shape for fc_loop: [batch_size, vertex_num, features]
        conv_out2 = conv_out2.permute(0, 2, 1)
        
        # Loop transformation with compatible input/output size
        loop = self.fc_loop(conv_out2)
        
        # Apply activation function
        return self.activation(loop)
