
import torch
from typing import Any, Dict
from torch_geometric.loader import DataLoader
from src.models.GradientPenalty import gradient_penalty
from src.utils import visualize_point_cloud


def train(generator: torch.nn.Module, 
          discriminator: torch.nn.Module, 
          dataloader: DataLoader, 
          epochs: int, 
          device: torch.device, 
          lr_g: float, 
          lr_d: float) -> None:
    """
    Trains the TreeGAN model.

    Args:
        generator (nn.Module): The generator model.
        discriminator (nn.Module): The discriminator model.
        dataloader (DataLoader): DataLoader containing the training data.
        epochs (int): Number of training epochs.
        device (torch.device): Device on which the model is trained (CPU or GPU).
        lr_g (float): Learning rate for the generator.
        lr_d (float): Learning rate for the discriminator.

    Returns:
        None
    """
    # Initialize optimizers for generator and discriminator
    opt_G = torch.optim.Adam(generator.parameters(), lr=lr_g)
    opt_D = torch.optim.Adam(discriminator.parameters(), lr=lr_d)

    # Logging dictionary to store losses
    loss_log: Dict[str, Any] = {'G_loss': [], 'D_loss': []}

    for epoch in range(epochs):
        for i, data in enumerate(dataloader):
            real_point_clouds = data.pos.to(device)  # Move real data to device

            # Training Discriminator
            z = torch.randn(real_point_clouds.size(0), 1, 96).to(device)  # Random noise for generator
            fake_point_clouds = generator(z)

            D_real = discriminator(real_point_clouds)  # Discriminator output for real data
            D_fake = discriminator(fake_point_clouds)  # Discriminator output for fake data
            
            # Compute Gradient Penalty
            gp_loss = gradient_penalty(discriminator, real_point_clouds, fake_point_clouds)
            d_loss = -D_real.mean() + D_fake.mean() + gp_loss
            
            # Update Discriminator
            opt_D.zero_grad()
            d_loss.backward()
            opt_D.step()

            # Training Generator
            fake_point_clouds = generator(z)  # Generate new fake point clouds
            D_fake = discriminator(fake_point_clouds)  # Discriminator output for new fake data
            g_loss = -D_fake.mean()  # Generator loss
            
            # Update Generator
            opt_G.zero_grad()
            g_loss.backward()
            opt_G.step()
            
            # Logging Losses
            loss_log['G_loss'].append(g_loss.item())
            loss_log['D_loss'].append(d_loss.item())
        
        # Print average losses for the epoch
        avg_g_loss = sum(loss_log['G_loss']) / len(loss_log['G_loss'])
        avg_d_loss = sum(loss_log['D_loss']) / len(loss_log['D_loss'])
        print(f'Epoch [{epoch + 1}/{epochs}] G_loss: {avg_g_loss:.4f} | D_loss: {avg_d_loss:.4f}')