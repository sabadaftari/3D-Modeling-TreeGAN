import torch.nn as nn

# Define the Discriminator
class TreeGANDiscriminator(nn.Module):
    """
    TreeGAN Discriminator for distinguishing between real and generated 3D point clouds.
    
    Args:
        features (int): Number of features of the input (3 for 3D points).
    
    Methods:
        forward(x): Passes the input through the discriminator and returns a score.
    """
    def __init__(self, features):
        super(TreeGANDiscriminator, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(features, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1)
        )
        print("here.")

    def forward(self, x):
        """ Passes the input point cloud `x` through the discriminator network to get the real/fake score. """
        return self.features(x).squeeze()
        # shape = x.shape[0]
        # if shape>= 60000:
        #     # Assuming your discriminator is defined as self.discriminator
        #     batch_size = 1024  # Define an appropriate batch size
        #     # Process in batches
        #     x_flat = x.view(-1, 3) 
        #     outputs = [self.features(x_flat[i:i + batch_size]) for i in range(0, shape, batch_size)]  # List to store outputs from each batch

        #     # Process in batches
        #     # for i in range(0, x.shape[0], batch_size):
        #     #     batch = x.view(-1, 3)[i:i + batch_size]
        #     #     output = self.features(batch)  # Get output for each batch
        #     #     outputs.append(output)  # Append the output to the list

        #     # Concatenate all outputs into a single tensor
        #     all_outputs = torch.cat(outputs, dim=0)
        #     return all_outputs.squeeze()
        # else:
        #     return self.features(x).squeeze()