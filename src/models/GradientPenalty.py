import torch

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
    # alpha = torch.rand(real_data.size(0), 1, 1).to(real_data.device)  # Random scalar for interpolation
    # interpolates = real_data + alpha * (fake_data - real_data)

    # Define the desired fraction for sampling
    # Define the desired fraction for sampling
    sampling_fraction = 0.05

    # Calculate the number of samples to use
    total_samples = real_data.size(0)  # Total number of samples in real_data
    # num_samples = int(total_samples * sampling_fraction)
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

    # x_flat = interpolates.view(-1, 3) 
    # dataset = TensorDataset(x_flat)  # Create a dataset
    # dataloader = DL(dataset, batch_size=1024, shuffle=False, num_workers=6) 
    # outputs = []
    # for batch in dataloader:
    #     batch_data = batch[0]  # Get the batch data
    #     output = discriminator(batch_data)  # Process the batch
    #     outputs.append(output)
    # # outputs = [discriminator(batch_data[0]) for batch_data in dataloader]
    # d_interpolates = torch.cat(outputs, dim=0)
    d_interpolates = discriminator(interpolates.view(-1, 3))
    
    gradients = torch.autograd.grad(
        outputs=d_interpolates, inputs=interpolates,
        grad_outputs=torch.ones(d_interpolates.size()).to(real_data.device),
        create_graph=True, retain_graph=True, only_inputs=True)[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
    # print("gradient finished.")
    return gradient_penalty