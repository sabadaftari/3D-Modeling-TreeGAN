import torch

# Gradient Penalty Calculation
def gradient_penalty(discriminator: torch.nn.Module, 
                     real_data: torch.Tensor, 
                     fake_data: torch.Tensor, 
                     lambda_gp: float = 10) -> torch.Tensor :
    
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

    # Calculate the number of samples to use, otherwise it would be too computationally expensive
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

    return gradient_penalty