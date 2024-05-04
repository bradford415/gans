# Usage: python scripts/train.py scripts/configs/base-config.yaml scripts/configs/dcgan/model-base.yaml
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import torch
import yaml
from fire import Fire
from torch import nn
from torch.utils.data import DataLoader

from gans.data.celeb_faces_a import build_CelebFacesA
from gans.models.dcgan import DCGenerator
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
    generator = backbone_map[model_config["backbone"]["name"]](
        pretrain=model_config["backbone"]["pretrained"],
        remove_top=model_config["backbone"]["remove_top"],
    )

    model_components = {"backbone": backbone, "num_classes": 80}

    # model = resnet50()  # Using temp resnet50 model
    # Initialize detection model
    model = detectors_map[model_config["detector"]](**model_components)
    criterion = nn.CrossEntropyLoss()

    # Extract the train arguments from base config
    train_args = {**base_config["train"]}

    # Initialize training objects
    optimizer, lr_scheduler = _init_training_objects(
        model_params=model.parameters(),
        optimizer=train_args["optimizer"],
        scheduler=train_args["scheduler"],
        learning_rate=train_args["learning_rate"],
        weight_decay=train_args["weight_decay"],
        lr_drop=train_args["lr_drop"],
    )

    runner = Trainer(output_path=base_config["output_path"])

    ## TODO: Implement checkpointing somewhere around here (or maybe in Trainer)

    # Build trainer args used for the training
    trainer_args = {
        "model": model,
        "criterion": criterion,
        "data_loader": dataloader_train,
        "optimizer": optimizer,
        "scheduler": lr_scheduler,
        "device": device,
        **train_args["epochs"],
    }
    runner.train(**trainer_args)


def _init_training_objects(
    model_params: Iterable,
    optimizer: str = "sgd",
    scheduler: str = "step_lr",
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    lr_drop: int = 200,
):
    optimizer = optimizer_map[optimizer](
        model_params, lr=learning_rate, weight_decay=weight_decay
    )
    lr_scheduler = scheduler_map[scheduler](optimizer, lr_drop)
    return optimizer, lr_scheduler


if __name__ == "__main__":
    Fire(main)
