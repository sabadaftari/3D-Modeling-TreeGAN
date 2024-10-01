import torch
import torch.nn as nn
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader as DL
import tracemalloc

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

# TreeGCN class to define the architecture of Tree-based Graph Neural Network
class TreeGCN(nn.Module):
    """
    Tree-based Graph Convolutional Network (TreeGCN) layer for generating 3D point clouds.
    
    Args:
        features (list): A list containing the number of features for each layer of the generator.
        degrees (list): A list containing the degree (number of child nodes) for each layer in the tree structure.
        layer_idx (int): The current layer index.
    
    Methods:
        forward(z): Processes the input `z` through root, loop transformations, and Conv1d, applying LeakyReLU activation.
    """
    def __init__(self, features, degrees, layer_idx):
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

    def forward(self, z):
        """ 
        Applies the TreeGCN transformation to the input `z`, performing both root, loop transformations, 
        and Conv1d layers for improved local feature extraction.
        """
        # Root transformation
        root = self.fc_root(z)
        
        # Reshape the output to fit Conv1d input: [batch_size, features, length]
        # root: [batch_size, vertex_num, features] -> [batch_size, features, vertex_num]
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



# Dataset loading
def get_dataloader(batch_size, dataset_path="src/data/ModelNet10"):
    """
    Loads the ModelNet10 dataset and returns a DataLoader.
    
    Args:
        batch_size (int): Batch size for training.
        dataset_path (str): Path to the dataset directory.
        
    Returns:
        dataloader (DataLoader): A DataLoader for the ModelNet dataset.
    """
    dataset = ModelNet(root=dataset_path, name='10', train=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=6)
    return dataloader

# Training TreeGAN Model
def train_treegan(generator, discriminator, dataloader, epochs, device):
    """
    Trains the TreeGAN model.
    
    Args:
        generator (nn.Module): The generator model.
        discriminator (nn.Module): The discriminator model.
        dataloader (DataLoader): DataLoader containing the training data.
        epochs (int): Number of training epochs.
        device (torch.device): Device on which the model is trained (CPU or GPU).
    
    Returns:
        None
    """
    opt_G = torch.optim.Adam(generator.parameters(), lr=0.0001)
    opt_D = torch.optim.Adam(discriminator.parameters(), lr=0.0001)
    loss_log = {'G_loss': [], 'D_loss': []}
    
    for epoch in range(epochs):
        for data in dataloader:
            real_point_clouds = data.pos.to(device)
            
            # Training Discriminator
            # print("training discriminator...")
            z = torch.randn(real_point_clouds.size(0), 1, 96).to(device)  # Random noise for generator
            fake_point_clouds = generator(z)
            
            D_real = discriminator(real_point_clouds)  # Discriminator on real data
            D_fake = discriminator(fake_point_clouds)  # Discriminator on generated (fake) data
            
            # Compute Gradient Penalty
            gp_loss = gradient_penalty(discriminator, real_point_clouds, fake_point_clouds)
            d_loss = -D_real.mean() + D_fake.mean() + gp_loss
            
            # Update Discriminator
            opt_D.zero_grad()
            d_loss.backward()
            opt_D.step()
            # print("Training Generator...")
            # Training Generator
            fake_point_clouds = generator(z)
            D_fake = discriminator(fake_point_clouds)
            g_loss = -D_fake.mean()
            
            # Update Generator
            opt_G.zero_grad()
            g_loss.backward()
            opt_G.step()
            
            # Logging Losses
            loss_log['G_loss'].append(g_loss.item())
            loss_log['D_loss'].append(d_loss.item())
            
        print(f'Epoch [{epoch}/{epochs}] G_loss: {sum(loss_log["G_loss"])/len(loss_log["G_loss"])} | D_loss: {sum(loss_log["D_loss"])/len(loss_log["D_loss"])}')

# Gradient Penalty Calculation
def gradient_penalty(discriminator, real_data, fake_data, lambda_gp=10):
    """
    Computes the Gradient Penalty for stabilizing the GAN training.
    
    Args:
        discriminator (nn.Module): The discriminator model.
        real_data (torch.Tensor): The real point cloud data.
        fake_data (torch.Tensor): The generated (fake) point cloud data.
        lambda_gp (float): Gradient penalty coefficient.
        
    Returns:
        gradient_penalty (torch.Tensor): The computed gradient penalty.
    """

    # Calculate the number of samples to use
    total_samples = real_data.size(0)  # Total number of samples in real_data

    num_samples = 1024
    # Randomly sample indices
    random_indices = torch.randperm(total_samples)[:num_samples]

    # Sample real and fake data
    sampled_real_data = real_data[random_indices]
    sampled_fake_data = fake_data[random_indices]

    # Ensure alpha is the same size as the sampled data
    alpha = torch.rand(num_samples, 1, 1, requires_grad=True).to(real_data.device)

    # Randomly mix sampled real and fake data
    interpolates = sampled_real_data + alpha * (sampled_fake_data - sampled_real_data)


    interpolates.requires_grad_(True)
    d_interpolates = discriminator(interpolates.view(-1, 3))
    
    gradients = torch.autograd.grad(
        outputs=d_interpolates, inputs=interpolates,
        grad_outputs=torch.ones(d_interpolates.size()).to(real_data.device),
        create_graph=True, retain_graph=True, only_inputs=True)[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
    # print("gradient finished.")
    return gradient_penalty


# Main Function to Run the Model
if __name__ == '__main__':
    # Store 25 frames
    tracemalloc.start(25)

    # Set fixed random number seed
    torch.manual_seed(1234)
    torch.set_printoptions(sci_mode=False)

    # clear cuda cache
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision("medium")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define the architecture of TreeGAN
    features = [96, 64, 64, 64, 3]  # Feature dimensions
    degrees = [2, 2, 2, 64]  # Tree node degrees
    batch_size = 10
    
    # Initialize the Generator and Discriminator
    generator = TreeGANGenerator(features, degrees, batch_size).to(device)
    discriminator = TreeGANDiscriminator(features[-1]).to(device)
    from src.loaders import Point_DataLoader
    # Load the data from ModelNet
    dataloader, valid = Point_DataLoader(batch_size)
    
    # Train TreeGAN
    train_treegan(generator, discriminator, dataloader, epochs=3, device=device)
    
    # After training
    from src.workflow.validate import evaluate_generator
    for valbatch in valid:
        evaluate_generator(generator, valbatch, device, num_samples=5)

