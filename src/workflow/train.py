
import torch
from src.models.GradientPenalty import gradient_penalty
from src.utils import visualize_point_cloud

# Training TreeGAN Model
def train(generator, discriminator, dataloader, epochs, device, lr_g, lr_d):
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
    opt_G = torch.optim.Adam(generator.parameters(), lr=lr_g)
    opt_D = torch.optim.Adam(discriminator.parameters(), lr=lr_d)
    loss_log = {'G_loss': [], 'D_loss': []}
    i = 0
    for epoch in range(epochs):
        for data in dataloader:
            if i<=1:
                real_point_clouds = data.pos.to(device)
                # visualize
                # visualize_point_cloud(real_point_clouds)
                # Training Discriminator
                print("training discriminator...")
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
                print("Training Generator...")
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
                i+=1
    
    print(f'Epoch [{epoch}/{epochs}] G_loss: {sum(loss_log["G_loss"])/len(loss_log["G_loss"])} | D_loss: {sum(loss_log["D_loss"])/len(loss_log["D_loss"])}')
