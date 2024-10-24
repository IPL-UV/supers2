import importlib
import pathlib
from typing import Literal, Union

import torch

from supers2.download import download_weights
from supers2.models.tricks import CNNHardConstraint


class CustomModel(torch.nn.Module):
    def __init__(self, SRmodel: torch.nn.Module, HardConstraint: torch.nn.Module):
        super(CustomModel, self).__init__()
        self.sr_model = SRmodel
        self.hard_constraint = HardConstraint

    def forward(self, x):
        sr = self.sr_model(x)
        return self.hard_constraint(x, sr)


def get_model_name(
    model_type: Literal["SR", "Fusionx2", "Fusionx4"],
    model_name: Literal["cnn", "swin", "mamba"] = "cnn",
    model_size: Literal[
        "lightweight", "small", "medium", "expanded", "large"
    ] = "expanded",
    model_loss: Literal["l1", "superloss", "adversarial"] = "l1",
    weights_path: Union[str, pathlib.Path, None] = None,
):
    # Set the weights path
    if weights_path is not None:
        weights_path = pathlib.Path(weights_path)
    else:
        raise ValueError("This feature is not implemented yet")

    # Select the SR model
    model = "%s__%s__%s__%s" % (model_type, model_name, model_size, model_loss)
    model_fullpath = weights_path / (model + ".pth")
    if not model_fullpath.exists():
        download_weights(model_fullpath)

    print(f"Weights [{model_type}]: '{model_fullpath}'")

    return model_fullpath


def load_model_parameters(model_name: str, model_size: str):
    if model_name == "diffusion":
        return {}

    # Dictionary mapping model names and sizes to corresponding functions/classes
    model_mapping = {
        "cnn": {
            "lightweight": "cnn_lightweight",
            "small": "cnn_small",
            "medium": "cnn_medium",
            "expanded": "cnn_expanded",
            "large": "cnn_large",
        },
        "swin": {
            "lightweight": "swin_lightweight",
            "small": "swin_small",
            "medium": "swin_medium",
            "expanded": "swin_expanded",
            "large": "swin_large",
        },
        "mamba": {
            "lightweight": "mamba_lightweight",
            "small": "mamba_small",
            "medium": "mamba_medium",
            "expanded": "mamba_expanded",
            "large": "mamba_large",
        },
    }

    # Check if the model name is valid
    if model_name not in model_mapping:
        raise ValueError(f"Invalid model name: {model_name}")

    # Check if the model size is valid for the given model
    if model_size not in model_mapping[model_name]:
        raise ValueError(f"Invalid model size: {model_size}")

    # Load the 'experiment' module from 'supers2'
    experiment_module = importlib.import_module("supers2.experiment")
    experiment_module = importlib.reload(experiment_module)

    # Get the function/class name based on the model name and size
    model_function_name = model_mapping[model_name][model_size]

    # Ensure the function/class exists in the module
    if not hasattr(experiment_module, model_function_name):
        raise ValueError(
            f"Model function '{model_function_name}' not found in supers2.experiment"
        )

    # Retrieve the function/class
    model_params = getattr(experiment_module, model_function_name)

    return model_params


def load_model(model_name: str, model_params: dict, device: str = "cpu", **kwargs):
    # Dictionary mapping model names to corresponding modules and classes
    model_mapping = {
        "cnn_legacy": ("supers2.models.cnn_legacy", "CNNSR_legacy"),
        "cnn": ("supers2.models.cnn", "CNNSR"),
        "swin": ("supers2.models.swin", "Swin2SR"),
        "mamba": ("supers2.models.mamba", "MambaSR"),
        "diffusion": ("supers2.models.diffusion", "SRLatentDiffusion"),
    }

    # Check if the model name is valid
    if model_name not in model_mapping:
        raise ValueError(f"Model '{model_name}' not found")

    # Get the module and class names
    module_name, class_name = model_mapping[model_name]

    # Load the module and class
    model_module = importlib.import_module(module_name)
    model_module = importlib.reload(model_module)
    model_class = getattr(model_module, class_name)

    # Instantiate the model
    if model_name == "diffusion":
        return model_class(device=device, **kwargs)
    else:
        return model_class(**model_params)


def load_fusionx2_model(
    model_name: str,
    model_size: str,
    model_loss: str,
    weights_path: Union[str, pathlib.Path],
):

    # Get the model snippet
    model_snippet = get_model_name(
        model_type="Fusionx2",
        model_name=model_name,
        model_size=model_size,
        model_loss=model_loss,
        weights_path=weights_path,
    )

    # Load the weights
    weights_data = torch.load(model_snippet, map_location=torch.device("cpu"))

    # remove hard_constraint
    for key in list(weights_data.keys()):
        if "hard_constraint" in key:
            weights_data.pop(key)

    # Load the model parameters
    model_params = load_model_parameters(model_name, model_size)

    # If model name is CNN change to CNN_legacy
    if model_name == "cnn":
        model_name = "cnn_legacy"

    # Load the model
    FusionX2 = load_model(model_name, model_params)
    FusionX2.load_state_dict(weights_data)
    FusionX2.eval()
    for param in FusionX2.parameters():
        param.requires_grad = False

    # Define the Hard Constraint
    hard_constraint = CNNHardConstraint(
        filter_method="butterworth",
        filter_hyperparameters={"order": 6},
        scale_factor=2,
        in_channels=6,
        out_channels=[0, 1, 2, 3, 4, 5],
    )
    hard_constraint.eval()
    for param in hard_constraint.parameters():
        param.requires_grad = False

    # Apply Model then hard constraint
    return CustomModel(SRmodel=FusionX2, HardConstraint=hard_constraint)


def load_fusionx4_model(
    model_name: str,
    model_size: str,
    model_loss: str,
    weights_path: Union[str, pathlib.Path],
):

    # Get the model snippet
    model_snippet = get_model_name(
        model_type="Fusionx4",
        model_name=model_name,
        model_size=model_size,
        model_loss=model_loss,
        weights_path=weights_path,
    )

    # Load the weights
    weights_data = torch.load(model_snippet, map_location=torch.device("cpu"))

    # remove hard_constraint
    for key in list(weights_data.keys()):
        if "hard_constraint" in key:
            weights_data.pop(key)

    # Load the model parameters
    model_params = load_model_parameters(model_name, model_size)

    # If model name is CNN change to CNN_legacy
    if model_name == "cnn":
        model_name = "cnn_legacy"

    # Load the model
    FusionX4 = load_model(model_name, model_params)
    FusionX4.load_state_dict(weights_data)
    FusionX4.eval()
    for param in FusionX4.parameters():
        param.requires_grad = False

    # Define the Hard Constraint
    hard_constraint = CNNHardConstraint(
        filter_method="butterworth",
        filter_hyperparameters={"order": 6},
        scale_factor=4,
        in_channels=6,
        out_channels=[0, 1, 2, 3, 4, 5],
    )
    hard_constraint.eval()
    for param in hard_constraint.parameters():
        param.requires_grad = False

    # Apply Model then hard constraint
    return CustomModel(SRmodel=FusionX4, HardConstraint=hard_constraint)


def load_srx4_model(
    model_name: str,
    model_size: str,
    model_loss: str,
    weights_path: Union[str, pathlib.Path],
    device: str = "cpu",
    **kwargs
):

    # Get the model snippet
    model_snippet = get_model_name(
        model_type="SR",
        model_name=model_name,
        model_size=model_size,
        model_loss=model_loss,
        weights_path=weights_path,
    )

    # Load the weights
    weights_data = torch.load(model_snippet, map_location=device)

    # remove hard_constraint (if exists) # TODO remove
    for key in list(weights_data.keys()):
        if "hard_constraint" in key:
            weights_data.pop(key)

    # Load the model parameters
    model_params = load_model_parameters(model_name, model_size)
    model_params["in_channels"] = 4
    model_params["out_channels"] = 4
    model_params["upscale"] = 4

    # Load the model
    SRX4 = load_model(model_name, model_params, kwargs)
    SRX4.load_state_dict(weights_data)
    SRX4.eval()
    for param in SRX4.parameters():
        param.requires_grad = False

    # Define the Hard Constraint
    hard_constraint = CNNHardConstraint(
        filter_method="butterworth",
        filter_hyperparameters={"order": 6},
        scale_factor=4,
        in_channels=4,
        out_channels=[0, 1, 2, 3],
    )
    hard_constraint.eval()
    for param in hard_constraint.parameters():
        param.requires_grad = False

    # Apply Model then hard constraint
    return CustomModel(SRmodel=SRX4, HardConstraint=hard_constraint)
