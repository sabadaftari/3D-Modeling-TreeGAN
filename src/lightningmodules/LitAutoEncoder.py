import torch
import lightning.pytorch as pl

from src.workflow.validate import evaluate_generator
from src.models import gradient_penalty
from src.utils import (
                        compute_diversity,
                        visualize_combined_point_clouds)


class GAN(pl.LightningModule):
  def __init__(self, generator, discriminator, live):
    super().__init__()
    self.generator = generator
    self.discriminator = discriminator
    self.live = live
    # Disable automatic optimization
    self.automatic_optimization = False
    # self.save_hyperparameters(ignore=['generator','discriminator'])
    self.generated_point_clouds = torch.empty(1,1)
    self.chamfer_loss = torch.empty(1,1)
    self.generated_samples = torch.empty(1,1)

  def forward(self, z):
    """
    Generates an image using the generator
    given input noise z
    """
    return self.generator(z)
  
  def configure_optimizers(self):
    opt_G = torch.optim.Adam(self.generator.parameters(), lr=0.0001)
    opt_D = torch.optim.Adam(self.discriminator.parameters(), lr=0.0001)
    return [opt_G, opt_D]

  def generator_step(self, x):
    """
    Training step for generator
    1. Sample random noise
    2. Pass noise to generator to
       generate images
    3. Classify generated images using
       the discriminator
    4. Backprop loss
    """
    
    fake_point_clouds = self.generator(x)
    D_fake = self.discriminator(fake_point_clouds)
    g_loss = -D_fake.mean()

    return g_loss

  def discriminator_step(self, x, z):
    """
    Training step for discriminator
    1. Get actual images
    2. Get fake images from generator
    3. Predict probabilities of actual images
    4. Predict probabilities of fake images
    5. Get loss of both and backprop
    """
    
    fake_point_clouds = self.generator(z)
    
    D_real = self.discriminator(x)  # Discriminator on real data
    D_fake = self.discriminator(fake_point_clouds)  # Discriminator on generated (fake) data
    
    # Compute Gradient Penalty
    gp_loss = gradient_penalty(self.discriminator, x, fake_point_clouds)
    d_loss = -D_real.mean() + D_fake.mean() + gp_loss

    return d_loss

  def training_step(self, batch, batch_idx):
    if batch_idx <=10:
        real_point_clouds = batch.pos
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        z = torch.randn(real_point_clouds.size(0), 1, 96).to(self.device) # Random noise for generator
        # Get the optimizers
        opt_G, opt_D = self.optimizers()
        # train discriminator
        
        d_loss = self.discriminator_step(real_point_clouds,z)
        # Update Discriminator
        opt_D.zero_grad()
        d_loss.backward()
        opt_D.step()

        # train generator
        
        g_loss = self.generator_step(z)
        # Update Generator
        opt_G.zero_grad()
        g_loss.backward()
        opt_G.step()
        
        return d_loss, g_loss

  def validation_step(self, valbatch, valbatch_idx):
    if valbatch_idx<=10:

        generated_point_clouds, chamfer_loss, generated_samples = evaluate_generator(self.generator, valbatch, self.device, num_samples=5)

        if valbatch_idx==0:
            self.generated_point_clouds = generated_point_clouds
            self.chamfer_loss = chamfer_loss
            self.generated_samples = generated_samples
        else:
            self.generated_point_clouds = torch.concat([self.generated_point_clouds,generated_point_clouds])
            self.chamfer_loss = torch.concat([self.chamfer_loss,chamfer_loss])
            self.generated_samples = torch.cat([self.generated_samples,generated_samples])

        return {"generated_point_clouds":generated_point_clouds,
                "chamfer_loss": chamfer_loss, 
                "generated_samples": generated_samples}
  
  def on_validation_batch_end(self,):
    # Visualize
    visualize_combined_point_clouds(self.generated_point_clouds)
    
    # Compute average Chamfer Distance
    avg_chamfer_loss = sum(self.chamfer_loss) / self.chamfer_loss

    # Compute Diversity
    diversity_score = compute_diversity(torch.cat(self.generated_samples, dim=0))

    print("blah")
  
  

#   def training_epoch_end(self, training_step_outputs):
#     epoch_test_images = self(self.test_noises)
#     self.test_progression.append(epoch_test_images)