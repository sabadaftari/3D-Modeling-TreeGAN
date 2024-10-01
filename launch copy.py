import torch
import tracemalloc
from src.models import (TreeGANGenerator,
                        TreeGANDiscriminator)
from src.workflow import train
from src.loaders import Point_DataLoader
from src.workflow.validate import evaluate_generator

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
    
    # Load the data from ModelNet
    dataloader, valid = Point_DataLoader(batch_size)
    
    # Train TreeGAN
    train(generator, discriminator, dataloader, epochs=3, device=device)
    
    # Evaluate TreeGAN
    for valbatch in valid:
        evaluate_generator(generator, valbatch, device, num_samples=5)

