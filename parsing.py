import argparse

def add_parse_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Adds command-line arguments to the provided parser for configuring the GAN training process.
    
    Args:
        parser (argparse.ArgumentParser): The argument parser instance to which the arguments will be added.
        
    Returns:
        argparse.ArgumentParser: The updated parser with added arguments.
    """
    # Metadata arguments
    parser.add_argument('--num_epochs', 
                        type=int, 
                        default=10,
                        help='Number of training epochs.')
    
    parser.add_argument('--batch_size', 
                        type=int, 
                        default=10, 
                        help='Size of each training batch.')
    
    parser.add_argument('--g_lr', 
                        type=float, 
                        default=0.0001, 
                        help='Learning rate for the generator.')
    
    parser.add_argument('--d_lr', 
                        type=float, 
                        default=0.0001, 
                        help='Learning rate for the discriminator.')
    
    parser.add_argument('--features', 
                        type=int, 
                        default=[96, 64, 64, 64, 3], 
                        nargs='+', 
                        help='Feature dimensions for the generator architecture.')
    
    parser.add_argument('--degrees', 
                        type=int, 
                        default=[2, 2, 2, 64], 
                        nargs='+', 
                        help='Degrees of tree nodes for the generator architecture.')
    
    return parser
