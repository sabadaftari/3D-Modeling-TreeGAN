
def add_parse_args(parser):
    # Metadata args
    parser.add_argument('--num_epochs', 
                            type = int, 
                            default = 1,
                            )
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--g_lr', default = 0.0001, type=float)
    parser.add_argument('--d_lr', default = 0.0001, type=float)
    parser.add_argument('--features', default = [96, 64, 64, 64, 3], type=int, nargs='+', help = '''Feature dimensions''')
    parser.add_argument('--degrees', default = [2, 2, 2, 64], type=int, nargs='+', help = '''Tree node degrees''')
    
    return parser