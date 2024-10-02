import torch
from typing import Tuple, Any
from src.utils import (ChamferDistance,
                        compute_diversity,
                        visualize_combined_point_clouds)
from src.models import gradient_penalty                   

def evaluate_generator(discriminator: torch.nn.Module, 
                       generator: torch.nn.Module, 
                       batch: Any, 
                       device: torch.device) -> Tuple[torch.Tensor, float]:
    """
    Evaluates the generator using the following metrics:
    1. Visual Inspection (using matplotlib)
    2. Chamfer Distance between generated and real point clouds.
    
    Args:
        discriminator (nn.Module): The discriminator model.
        generator (nn.Module): The generator model.
        dataloader (DataLoader): DataLoader containing the validation data.
        device (torch.device): Device for computation (CPU or GPU).
        
    Returns:
        chamfer_loss (float): Average Chamfer distance between real and generated point clouds.
        loss (float): The negative of the average score given by the discriminator for the generated point clouds. 
        A higher score indicates that the discriminator believes the samples are more likely real.
    """
    generator.eval()  # Set generator to evaluation mode
    chamfer_distances = []
    
    with torch.no_grad():
        real_point_clouds = batch.pos.to(device)  # Get the real point clouds
        batch_size = real_point_clouds.size(0)

        # Generate point clouds using the generator
        z = torch.randn(batch_size, 1, 96).to(device)  # Generate random noise
        generated_point_clouds = generator(z)

        # Compute Chamfer Distance
        # chamfer_loss = ChamferDistance(real_point_clouds, generated_point_clouds)
        # chamfer_distances.append(chamfer_loss.item())

        D_fake = discriminator(generated_point_clouds)
        g_loss = -D_fake.mean()

    return generated_point_clouds, g_loss