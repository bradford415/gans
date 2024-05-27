import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from gans.evaulation import plots
from gans.utils import misc_utils


class Trainer:
    """Trainer TODO: comment"""

    def __init__(self, exp_name: str, log_train_steps: int = 500):
        """Constructor for the Trainer class

        Args:
            TODO
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {self.device}")

        ## TODO: PROBALBY REMOVE THESE Initialize training objects
        # self.optimizer = optimizer_map[optimizer]
        # self.lr_scheduler = "test"

        # Paths
        # self.output_paths = {
        #     "output_dir": Path(
        #         f"{output_path}_{datetime.now().strftime('%Y_%m_%d-%I_%M_%S_%p')}"
        #     ),
        # }
        self.output_dir = Path(
            f"output/{exp_name}/{datetime.now().strftime('%Y_%m_%d-%I_%M_%S_%p')}"
        )

        # Logging attributes
        self.log_train_steps = log_train_steps
        self.train_stats = {"disc_losses": [], "gen_losses": []}
        self.fixed_images = []

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
    ) -> dict[str, float]:
        """Trains one epoch

        Args:
            model_generator: _description_
            model_discriminator: _description_
            criterion: _description_
            data_loader: _description_
            gen_optimizer: _description_
            disc_optimizer: _description_
            fixed_noise: _description_
            input_noise_size: _description_. Defaults to 100.
            device: _description_. Defaults to torch.device("cpu").
            log_every_steps:
        """
        for steps, samples in enumerate(data_loader):
            samples = samples.to(device)

            # (Step 1) Train the discriminator: maximize log(D(x)) + log(1 - D(G(z)))
            disc_optimizer.zero_grad()

            # Create labels for the real image; by convention, real = 1 and fake = 0
            real_labels = torch.full(
                (samples.shape[0],), 1.0, dtype=torch.float, device=device
            )

            # Train the discriminator on real images and calculate error
            disc_logits = model_discriminator(samples)
            real_disc_loss = criterion(
                disc_logits.view(-1), real_labels
            )  # .view(-1) converts shape from (B, 1, 1, 1) to (B)
            real_disc_loss.backward()  # log(D(x))

            # Generate fake images with random latent vectors from a normal distribution;
            # do not train the generator yet
            noise = torch.randn(
                samples.shape[0], input_noise_size, 1, 1, device=device
            )  # (B, input_noise_size, 1, 1)
            fake_images = model_generator(noise)

            # Classify fake images and calculate error
            # fake_labels = torch.full(
            #     (samples.shape[0],), 0.0, dtype=torch.float, device=device
            # )
            real_labels.fill_(0.0)
            # pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu118

            # Notes on .detach():
            #   .detach() is necessary because we only want the discriminator on the computational graph.
            #   Since fake_images was passed through the generator, if we don't call detach the generator will be on the computational graph
            #   and pytorch will track fake_disc_loss all the way back to the generator (we don't want that).
            #   Calling .deatch(), we will only be accumulating gradients from the discriminator. Calling model_generator.<layer_name>.weight.grad will
            #   shown None when using .detach() (this is what we want)
            disc_logits = model_discriminator(fake_images.detach())
            #fake_disc_loss = criterion(disc_logits.view(-1), fake_labels)
            fake_disc_loss = criterion(disc_logits.view(-1), real_labels)
            fake_disc_loss.backward()  # log(1 - D(G(z)))

            # Gather the final discriminator loss and update the discriminator model
            total_disc_loss = real_disc_loss + fake_disc_loss
            disc_optimizer.step()

            # (Step 2) Train the Generator: maximize log(D(G(z)))
            # A higher log loss means the discriminator thinks the fake image is real
            model_generator.zero_grad()

            # Even though we pass the discriminator fake labels, now that we are training the generator
            # In the original paper, minimizing loss log(1 - (D(G(z))) did not provide sufficient gradients
            # minimizing log(1 - (D(G(z))) means the discriminator thinks the fake image is real
            # As a fix, we want to now maximize (log(D(G(x)))) and this can be accomplished by giving the fake images
            # the "real" label when calculating the loss.
            # This works because if the discriminator thinks the fake image is real, it will have a high probablity such as .95
            # real_labels_2 = torch.full(
            #     (samples.shape[0],), 1.0, dtype=torch.float, device=device
            # )
            real_labels.fill_(1.0)
            fake_disc_logits = model_discriminator(fake_images)
            fake_gen_loss = criterion(fake_disc_logits.view(-1), real_labels)

            # Because we aren't using .detach anymore, backward() will track the loss from the discriminator all the way back to the generator
            fake_gen_loss.backward()

            # Update optimizer
            gen_optimizer.step()

            # Output training stats
            if steps % self.log_train_steps - 1 == 0:
                print(
                    f"\tSteps[{steps-1}/{len(data_loader)}]\tLoss_D: {total_disc_loss:.4f}\tLoss_G: {fake_gen_loss:.4f}\tD(x):"
                )

            # Save losses to plot later
            self.train_stats["disc_losses"].append(total_disc_loss.item())
            self.train_stats["gen_losses"].append(fake_gen_loss.item())

        # Generate the same images from fixed_noise at the end of every epoch to visualize the training progress; sigmoid to bound to [0, 1] (should consider putting sigmoid in model itself)
        with torch.no_grad():
            fixed_images = model_generator(fixed_noise).detach().cpu().numpy()
            #fixed_images = F.sigmoid(fixed_images)

        return fixed_images

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
        TODO
            model:
            optimizer:
            ckpt_every:
        """
        print(f"Outputs will be saved in {self.output_dir}")
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize fixed noise ONLY to visualize the progression of the generator;
        # we still feed the generator random vectors every training step
        fixed_noise = torch.rand(
            64, input_noise_size, 1, 1, device=device
        )  # (64, 100, 1, 1)

        print("Start training")
        start_time = time.time()
        for epoch in range(start_epoch, epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}:")
            fixed_images = self._train_one_epoch(
                model_generator,
                model_discriminator,
                criterion,
                data_loader,
                gen_optimizer,
                disc_optimizer,
                fixed_noise,
                input_noise_size,
                device,
            )

            visuals_dir = self.output_dir / "visuals" / "fixed_images"
            visuals_dir.mkdir(parents=True, exist_ok=True)
            plots.visualize_fixed_images(
                fixed_images, save_path=visuals_dir / f"epoch{epoch:03}.png"
            )

            # Save the model every ckpt_every
            if ckpt_every is not None and (epoch + 1) % ckpt_every == 0:
                print(f"Checkpointing model at epoch: {epoch + 1}. ")
                ckpt_disc_path = self.output_dir / f"disc_checkpoint{epoch:04}.pt"
                ckpt_gen_path = self.output_dir / f"gen_checkpoint{epoch:04}.pt"
                self._save_model(
                    model_discriminator,
                    disc_optimizer,
                    epoch,
                    save_path=ckpt_disc_path,
                )
                self._save_model(
                    model_generator,
                    gen_optimizer,
                    epoch,
                    save_path=ckpt_gen_path,
                )

        # Post training
        plots.plot_losses(
            self.train_stats["gen_losses"],
            self.train_stats["disc_losses"],
            save_path=self.output_dir / "visuals" / "losses.png",
        )

        print("Training finished. Outputs stored in")

    def _save_model(
        self,
        model: nn.Module,
        optimizer: nn.Module,
        current_epoch: int,
        save_path: str,
        lr_scheduler: Optional[nn.Module] = None,
    ):
        """TODO"""
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict()
                if lr_scheduler is not None
                else None,
                "epoch": current_epoch,
            },
            save_path,
        )
