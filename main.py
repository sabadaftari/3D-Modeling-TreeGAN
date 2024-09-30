import torch
import tracemalloc
import lightning.pytorch as pl
from src.models import (TreeGANGenerator,
                        TreeGANDiscriminator)

from src.loaders import Point_DataLoader
from src.lightningmodules import GAN

from dvclive.lightning import DVCLiveLogger
from dvclive import Live

if __name__ == '__main__':
    # Store 25 frames
    tracemalloc.start(25)

    # Set fixed random number seed
    torch.manual_seed(1234)
    torch.set_printoptions(sci_mode=False)

    # clear cuda cache
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision("medium")
    
    # Define the architecture of TreeGAN
    features = [96, 64, 64, 64, 3]  # Feature dimensions
    degrees = [2, 2, 2, 64]  # Tree node degrees
    batch_size = 10

    # Load the data from ModelNet
    dataloader = Point_DataLoader(batch_size)

    epochs = 20
    with Live("dvclive") as live:
        dvclive_logger = DVCLiveLogger(save_dvc_exp=True)
        # Initialize the Generator and Discriminator
        generator = TreeGANGenerator(features, degrees, batch_size)
        discriminator = TreeGANDiscriminator(features[-1])
        model = GAN(generator, discriminator,live)
        trainer = pl.Trainer(
            check_val_every_n_epoch=1,
            # devices=1, 
            accelerator="cpu",
            max_epochs=epochs,
            # default_root_dir=params["BASE_URL"],
            logger=dvclive_logger
        )

        """                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
        TRAIN & VALIDATE
        """
        # print(f">>>>>>>>>> Number of train batches:  {len(loader.train)}")
        # print(f">>>>>>>>>> Number of validation batches:  {len(loader.validate)}")
        
        # Calling the train and validate
        trainer.fit(model, dataloader)
        # trainer.fit(model, loader.entire_train) #once your model is ready + hparams are tuned, we train on the total dataset
        """
        TEST
        """
        # print(f">>>>>>>>>> Number of test batches:  {len(loader.test)}")
        # # Calling the test with checkpoint from an already trained model
        # y_hat_all_val = trainer.test(model, 
        #                             dataloaders=loader.test, 
        #                             ckpt_path=params["CHECKPOINT_URL"]) # using trainer.test will perform one evaluation epoch over the dataloader given.

    