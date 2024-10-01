import torch
from src.utils import (ChamferDistance,
                        compute_diversity,
                        visualize_combined_point_clouds)
                   

def evaluate_generator(generator, batch, device, num_samples=5):
    """
    Evaluates the generator using the following metrics:
    1. Visual Inspection (using matplotlib)
    2. Chamfer Distance between generated and real point clouds.
    3. Diversity of generated samples.
    
    Args:
        generator (nn.Module): The generator model.
        dataloader (DataLoader): DataLoader containing the validation data.
        device (torch.device): Device for computation (CPU or GPU).
        num_samples (int): Number of generated samples for visual inspection.
        
    Returns:
        chamfer_loss (float): Average Chamfer distance between real and generated point clouds.
        diversity (float): Diversity score based on Chamfer distance between generated samples.
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


    return generated_point_clouds