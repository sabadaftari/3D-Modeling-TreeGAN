import torch
import tracemalloc
import argparse
from src.loaders import Point_DataLoader
from src.models import (TreeGANGenerator,
                        TreeGANDiscriminator)
from src.utils import visualize_point_cloud, visualize_batch_output_single_plot
from src.workflow import (train, evaluate_generator)
from parsing import add_parse_args

# Store 25 frames
tracemalloc.start(25)

# Set fixed random number seed
torch.manual_seed(1234)
torch.set_printoptions(sci_mode=False)

# clear cuda cache
torch.cuda.empty_cache()
torch.set_float32_matmul_precision("medium")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Main Function to Run the Model
def main():

    """
    READ ARGUMENTS
    """
    # First let's read args for our experiment
    parser = argparse.ArgumentParser()
    parser = add_parse_args(parser)
    args = parser.parse_args()
    
    # Initialize the Generator and Discriminator
    generator = TreeGANGenerator(args.features, args.degrees, args.batch_size).to(device)
    discriminator = TreeGANDiscriminator(args.features[-1]).to(device)
    
    # Load the data from ModelNet
    dataloader, valid = Point_DataLoader(args.batch_size)
    
    # Train TreeGAN
    train(generator, 
          discriminator, 
          dataloader, 
          epochs=args.num_epochs, 
          device=device, 
          lr_g= args.g_lr, 
          lr_d = args.d_lr)
    
    # Evaluate TreeGAN
    for valbatch in valid:
        visualize_point_cloud(valbatch.pos.to(device))
        generated_point_clouds = evaluate_generator(generator, valbatch, device, num_samples=5)
        visualize_batch_output_single_plot(generated_point_clouds)

if __name__ == '__main__':
    main()