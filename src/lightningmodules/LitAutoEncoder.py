import torch
import lightning.pytorch as pl

from src.workflow import train
from src.models import gradient_penalty


class GAN(pl.LightningModule):
  def __init__(self, generator, discriminator, live):
    super().__init__()
    self.generator = generator
    self.discriminator = discriminator
    # Disable automatic optimization
    self.automatic_optimization = False

  def forward(self, z):
    """
    Generates an image using the generator
    given input noise z
    """
    return self.generator(z)

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

    # Update Generator
    self.opt_G.zero_grad()
    g_loss.backward()
    self.opt_G.step()

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

    # Update Discriminator
    self.opt_D.zero_grad()
    d_loss.backward()
    self.opt_D.step()

    return d_loss

  def training_step(self, batch, batch_idx):
    real_point_clouds = batch.pos
    z = torch.randn(real_point_clouds.size(0), 1, 96) # Random noise for generator

    # train discriminator
    
    loss = self.discriminator_step(real_point_clouds,z)

    # train generator
    
    loss = self.generator_step(z)

    return loss

  def configure_optimizers(self):
    opt_G = torch.optim.Adam(self.generator.parameters(), lr=0.0001)
    opt_D = torch.optim.Adam(self.discriminator.parameters(), lr=0.0001)
    return [opt_G, opt_D], []

#   def training_epoch_end(self, training_step_outputs):
#     epoch_test_images = self(self.test_noises)
#     self.test_progression.append(epoch_test_images)