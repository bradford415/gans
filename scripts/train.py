# Usage: python scripts/train.py scripts/configs/base-config.yaml scripts/configs/dcgan/model-base.yaml
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
import yaml
from fire import Fire
from torch import nn
from torch.utils.data import DataLoader

from gans.data.celeb_faces_a import build_CelebFacesA
from gans.models.dcgan import DCDiscriminator, DCGenerator
from gans.trainer import Trainer
from gans.utils import misc_utils

dataset_map: Dict[str, Any] = {"CelebFacesA": build_CelebFacesA}

optimizer_map = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
}

scheduler_map = {"step_lr": torch.optim.lr_scheduler.StepLR}

generator_map = {"dcgan_gen": DCGenerator}
discriminator_map = {"dcgan_disc": DCDiscriminator}


def main(base_config_path: str, model_config_path: str):
    """Entrypoint for the project

    Args:
        base_config_path: path to the desired configuration file
        model_config_path: path to the detection model configuration file
    """

    print("Initializations...\n")

    with open(base_config_path, "r") as f:
        base_config = yaml.safe_load(f)

    with open(model_config_path, "r") as f:
        model_config = yaml.safe_load(f)

    # Apply reproducibility seeds
    misc_utils.reproducibility(**base_config["reproducibility"])

    # Set cuda parameters
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    train_kwargs = {"batch_size": base_config["train"]["batch_size"], "shuffle": True}
    val_kwargs = {
        "batch_size": base_config["validation"]["batch_size"],
        "shuffle": False,
    }

    if use_cuda:
        print(f"Using {len(base_config['cuda']['gpus'])} GPU(s): ")
        for gpu in range(len(base_config["cuda"]["gpus"])):
            print(f"    -{torch.cuda.get_device_name(gpu)}")
        cuda_kwargs = {
            "pin_memory": True,
        }

        train_kwargs.update(cuda_kwargs)
        val_kwargs.update(cuda_kwargs)
    else:
        print("Using CPU")
    dataset_kwargs = base_config["dataset"]
    dataset_train = dataset_map[base_config["dataset_name"]](
        dataset_split="train", **dataset_kwargs
    )

    dataloader_train = DataLoader(
        dataset_train,
        num_workers=base_config["cpu"]["num_workers"],
        **train_kwargs,
    )

    # Initalize model components
    model_generator = generator_map[model_config["gen_name"]](
        **model_config["generator"]
    )
    model_discriminator = discriminator_map[model_config["disc_name"]](
        **model_config["discriminator"]
    )

    model_generator = model_generator.to(device)
    model_discriminator = model_discriminator.to(device)

    # model_components = {"backbone": backbone, "num_classes": 80}

    criterion = nn.BCELoss()

    # Extract the train arguments from base config
    train_args = {**base_config["train"]}

    # Initialize training objects
    opt_name = train_args["optimizer_name"]
    optimizer_args = {
        "optimizer": opt_name,
        "learning_rate": train_args["learning_rate"],
        "weight_decay": train_args["weight_decay"],
        "opt_params": train_args["optimizers"][opt_name],
    }
    gen_optimizer, disc_optimizer = _init_training_objects(
        gen_params=model_generator.parameters(),
        disc_params=model_discriminator.parameters(),
        **optimizer_args,
    )

    runner = Trainer(
        exp_name=base_config["exp_name"], **base_config["logging"]
    )  # TODO: Consider building a nice logger class or some logging method

    # Build trainer args used for the training
    trainer_args = {
        "model_generator": model_generator,
        "model_discriminator": model_discriminator,
        "criterion": criterion,
        "data_loader": dataloader_train,
        "gen_optimizer": gen_optimizer,
        "disc_optimizer": disc_optimizer,
        "input_noise_size": model_config["generator"]["input_vec_ch"],
        "device": device,
        **train_args["epochs"],
    }
    runner.train(**trainer_args)


def _init_training_objects(
    gen_params: Iterable,
    disc_params: Iterable,
    optimizer: str = "sgd",
    learning_rate: float = 1e-4,
    weight_decay: float = 0.0,
    opt_params: dict | None = None,
):
    """TODO

    Args:

    Returns:

    """
    gen_optimizer = optimizer_map[optimizer](
        gen_params, lr=learning_rate, weight_decay=weight_decay, **opt_params
    )

    disc_optimizer = optimizer_map[optimizer](
        disc_params, lr=learning_rate, weight_decay=weight_decay, **opt_params
    )

    return gen_optimizer, disc_optimizer


if __name__ == "__main__":
    Fire(main)
