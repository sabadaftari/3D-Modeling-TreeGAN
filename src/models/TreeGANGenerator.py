import torch.nn as nn
from src.models import TreeGCN
# Define the Generator
class TreeGANGenerator(nn.Module):
    def __init__(self, features, degrees, batch_size):
        super(TreeGANGenerator, self).__init__()
        self.batch_size = batch_size
        self.layer_num = len(features) - 1
        self.vertex_num = 1  # Starting with 1 vertex
        self.gcn = nn.Sequential()

        # Create a TreeGCN for each layer in the generator
        for inx in range(self.layer_num):
            self.gcn.add_module(f'TreeGCN_{inx}', TreeGCN(features, degrees, inx))
            self.vertex_num *= degrees[inx]  # Update vertex_num for the next layer

        # Final layer to map to 3D coordinates (x, y, z)
        self.fc_final = nn.Linear(features[-1], 3)  # Last feature size to 3 (for x, y, z)

    def forward(self, z):
        """ Passes the noise vector `z` through the GCN layers to generate the 3D point cloud. """
        feat = self.gcn(z)
        # Apply the final fully connected layer to match the 3D output
        final_output = self.fc_final(feat)
        return final_output.view(-1, 3)  # Reshape to [total_points, 3]


