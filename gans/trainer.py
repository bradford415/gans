import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from gans.utils import misc_utils


class Trainer:
    """Trainer TODO: comment"""

    def __init__(self, output_path: str):
        """Constructor for the Trainer class

        Args:
            c
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {self.device}")

        ## TODO: PROBALBY REMOVE THESE Initialize training objects
        # self.optimizer = optimizer_map[optimizer]
        # self.lr_scheduler = "test"

        # Paths
        self.output_paths = {
            "output_dir": Path(
                f"{output_path}_{datetime.now().strftime('%Y_%m_%d-%I_%M_%S_%p')}"
            ),
        }

    def _train_one_epoch(
        self,
        model_generator: nn.Module,
        model_discriminator: nn.Module,
        criterion: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        gen_optimizer: torch.optim,
        disc_optimizer: torch.optim,
        fixed_noise: torch.Tensor,
        input_noise_size: int = 100,
        device: torch.device = torch.device("cpu"),
    ):
        for steps, (samples) in enumerate(tqdm(data_loader, ascii=" >=")):
            samples = samples.to(device)

            # Step 1: Train the discriminator: maximize log(D(x)) + log(1 - D(G(z)))
            disc_optimizer.zero_grad()
            
            # Create labels for the real image; by convention, real = 1 and fake = 0
            real_labels = torch.full((samples.shape[0], 1), 1.0, dtype=torch.float, device=device)
            
            # Classify real images and calculate error
            disc_logits = model_discriminator(samples)
            real_disc_loss = criterion(disc_logits, real_labels)
            real_disc_loss.backward() # log(D(x))

            # Train the generator with fake images        
            # Generate fake images with random latent vectors
            noise = torch.randn(samples.shape[0], input_noise_size, 1, 1, device=device)
            fake_images = model_generator(noise)
            
            # Classify fake images and calculate error
            fake_labels = torch.full((samples.shape[0], 1), 0.0, dtype=torch.float, device=device)
            disc_logits = model_discriminator(fake_images)
            fake_disc_loss = criterion(disc_logits, fake_labels)
            fake_disc_loss.backward() # log(1 - D(G(z)))

            # Gather the final discriminator loss and update the model
            total_disc_loss = real_disc_loss + fake_disc_loss
            disc_optimizer.step()
            ############### START HEERE, run code and figure out errors, shapes, and compare batch sizes from real and fake ################
            breakpoint()


            # Step 2: Train the
            exit()

            ## TODO: understand this and rename variables if needed
            loss, loss_xy, loss_wh, loss_obj, loss_cls, lossl2 = criterion(
                bbox_predictions, targets["bboxes"]
            )


    def train(
        self,
        model_generator: nn.Module,
        model_discriminator: nn.Module,
        criterion: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        gen_optimizer: torch.optim,
        disc_optimizer: torch.optim,
        input_noise_size: int = 100,
        start_epoch: int = 0,
        epochs: int = 100,
        ckpt_every: int = None,
        device: torch.device = torch.device("cpu"),
    ):
        """Train a model

        Args:
            model:
            optimizer:
            ckpt_every:
        """
        
        # Initialize fixed noise ONLY to visualize the progression of the generator; 
        # we still feed the generator random vectors every training step
        fixed_noise = torch.rand(64, input_noise_size, 1, 1, device=device)
        
        print("Start training")
        start_time = time.time()
        for epoch in range(start_epoch, epochs):
            train_stats = self._train_one_epoch(
                model_generator, model_discriminator, criterion, data_loader, gen_optimizer, disc_optimizer,fixed_noise, input_noise_size, device
            )

            # Save the model every ckpt_every
            if ckpt_every is not None and (epoch + 1) % ckpt_every == 0:
                ckpt_path = self.output_paths["output_dir"] / f"checkpoint{epoch:04}"
                self._save_model(
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    ckpt_every,
                    save_path=ckpt_path,
                )

    def _save_model(
        self, model, optimizer, lr_scheduler, current_epoch, ckpt_every, save_path
    ):
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": current_epoch,
            },
            save_path,
        )

    @torch.no_grad()
    def estimate_loss(
        self, train_data, val_data, model, eval_iters, batch_size, block_size
    ) -> Dict[str, float]:
        """Estimate the loss of the train and val split

        Args:

        """
        out = {}
        model.eval()
        all_data = {"train": train_data, "val": val_data}
        for split in ["train", "val"]:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = Vocab.get_batch(
                    all_data[split], batch_size, block_size, self.device
                )
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out
