import torch
import tracemalloc
import argparse
from src.loaders import Point_DataLoader
from src.models import (TreeGANGenerator,
                        TreeGANDiscriminator)
from src.utils import visualize_point_cloud, visualize_batch_output_single_plot
from src.workflow import (train, evaluate_generator)
from parsing import add_parse_args

# Store 25 frames that limits the number of stack frames stored during memory allocation tracking in Python
tracemalloc.start(25)

# Set fixed random number seed so we can generate same results every time we run
torch.manual_seed(1234)

# Not seeing numbers on terminal in scientific mode (optional)
torch.set_printoptions(sci_mode=False)

# Clear cuda cache
torch.cuda.empty_cache()

# Change precision to avoid OOM
torch.set_float32_matmul_precision("medium")

# Detect the devices on the machine
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Main Function to Run the Model
def main() -> None:
    """
    READ ARGUMENTS
    """
    # First let's read args for our experiment
    parser = argparse.ArgumentParser()
    parser = add_parse_args(parser)
    args = parser.parse_args()
    
    """
    CREATE DATALOADER
    """
    # Load the data from ModelNet
    dataloader, valid = Point_DataLoader(args.batch_size)

    """
    INITIAL MODELS
    """
    # Initialize the Generator and Discriminator
    generator = TreeGANGenerator(args.features, args.degrees, args.batch_size).to(device)
    discriminator = TreeGANDiscriminator(args.features[-1]).to(device)
    
    """
    TRAIN
    """
    # Train TreeGAN
    train(generator, 
          discriminator, 
          dataloader, 
          epochs=args.num_epochs, 
          device=device, 
          lr_g= args.g_lr, 
          lr_d = args.d_lr)
    
    """
    VALIDATE
    """
    loss = {'G_loss': []}
    # Evaluate Generator
    for valbatch in valid:
        visualize_point_cloud(valbatch.pos.to(device)) # visualize input data
        generated_point_clouds, loss = evaluate_generator(discriminator, generator, valbatch, device)
        visualize_batch_output_single_plot(generated_point_clouds) # visualize generated data
        loss['G_loss'].append(loss.item())
    print(f'G_loss: {sum(loss["G_loss"])/len(loss["G_loss"])}')


if __name__ == '__main__':
    main()