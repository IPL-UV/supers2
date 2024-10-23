import pathlib
from typing import Literal, Optional, Union

import torch

from supers2.setup import (load_fusionx2_model, load_fusionx4_model,
                           load_srx4_model)


def setmodel(
    resolution: Literal["2.5m", "5m", "10m"] = "2.5m",
    SR_model_name: Literal["cnn", "swin", "mamba"] = "cnn",
    SR_model_size: Literal[
        "lightweight", "small", "medium", "expanded", "large"
    ] = "small",
    SR_model_loss: Literal["l1", "superloss", "adversarial"] = "l1",
    Fusionx2_model_name: Literal["cnn", "swin", "mamba"] = "cnn",
    Fusionx2_model_size: Literal[
        "lightweight", "small", "medium", "expanded", "large"
    ] = "lightweight",
    Fusionx4_model_name: Literal["cnn", "swin", "mamba"] = "cnn",
    Fusionx4_model_size: Literal[
        "lightweight", "small", "medium", "expanded", "large"
    ] = "lightweight",
    weights_path: Union[str, pathlib.Path, None] = None,
) -> dict:
    """
    Sets up models for super-resolution and fusion tasks based on the specified parameters.

    Args:
        resolution (Literal["2.5m", "5m", "10m"], optional):
            Target spatial resolution. Determines which models to load.
            Defaults to "2.5m".
        SR_model_name (Literal["cnn", "swin", "mamba"], optional):
            The super-resolution model to use.
            Options: "cnn", "swin", "mamba". Defaults to "cnn".
        SR_model_size (Literal["lightweight", "small", "medium", "expanded", "large"], optional):
            Size of the super-resolution model.
            Options: "lightweight", "small", "medium", "expanded", "large".
            Defaults to "small".
        SR_model_loss (Literal["l1", "superloss", "adversarial"], optional):
            Loss function used in training the super-resolution model.
            Options: "l1", "superloss", "adversarial". Defaults to "l1".
        Fusionx2_model_name (Literal["cnn", "swin", "mamba"], optional):
            Model for Fusion X2 (e.g., 20m -> 10m resolution).
            Options: "cnn", "swin", "mamba". Defaults to "cnn".
        Fusionx2_model_size (Literal["lightweight", "small", "medium", "expanded", "large"], optional):
            Size of the Fusion X2 model.
            Options: "lightweight", "small", "medium", "expanded", "large".
            Defaults to "lightweight".
        Fusionx4_model_name (Literal["cnn", "swin", "mamba"], optional):
            Model for Fusion X4 (e.g., 10m -> 2.5m resolution).
            Options: "cnn", "swin", "mamba". Defaults to "cnn".
        Fusionx4_model_size (Literal["lightweight", "small", "medium", "expanded", "large"], optional):
            Size of the Fusion X4 model.
            Options: "lightweight", "small", "medium", "expanded", "large".
            Defaults to "lightweight".
        weights_path (Union[str, pathlib.Path, None], optional):
            Path to the pre-trained model weights.
            Can be a string or pathlib.Path object. Defaults to None.
            If None, the code will try to retrieve the weights from the
            official repository.

    Returns:
        dict: A dictionary containing the loaded models for super-resolution and fusion tasks.
            - "FusionX2": Loaded Fusion X2 model (or None if not used).
            - "FusionX4": Loaded Fusion X4 model (or None if not used).
            - "SR": Loaded super-resolution model (or None if not used).
    """
    # If weights_path is None, we create a folder in the .config directory
    if weights_path is None:
        weights_path = pathlib.Path.home() / ".config" / "supers2"
        weights_path.mkdir(parents=True, exist_ok=True)

    # If the resolution is 10m we only run the FusionX2 model that
    # converts 20m bands to 10m
    if resolution == 10:
        return {
            "FusionX2": load_fusionx2_model(
                model_name=Fusionx2_model_name,
                model_size=Fusionx2_model_size,
                model_loss="l1",
                weights_path=weights_path,
            ),
            "FusionX4": None,
            "SR": None,
        }

    else:
        return {
            "FusionX2": load_fusionx2_model(
                model_name=Fusionx2_model_name,
                model_size=Fusionx2_model_size,
                model_loss="l1",
                weights_path=weights_path,
            ),
            "FusionX4": load_fusionx4_model(
                model_name=Fusionx4_model_name,
                model_size=Fusionx4_model_size,
                model_loss="l1",
                weights_path=weights_path,
            ),
            "SR": load_srx4_model(
                model_name=SR_model_name,
                model_size=SR_model_size,
                model_loss=SR_model_loss,
                weights_path=weights_path,
            ),
        }


def predict(
    X: torch.Tensor,
    resolution: Literal["2.5m", "5m", "10m"] = "5m",
    models: Optional[dict] = None,
) -> torch.Tensor:
    """Generate a new S2 tensor with all the bands on the same resolution

    Args:
        X (torch.Tensor): The input tensor with the S2 bands
        resolution (Literal["2.5m", "5m", "10m"], optional): The final resolution of the
            tensor. Defaults to "2.5m".
        device (str, optional): The device to use. Defaults to "cpu".

    Returns:
        torch.Tensor: The tensor with the same resolution for all the bands
    """

    # Check if the models are loaded
    if models is None:
        models = setmodel(resolution=resolution)

    # if resolution is 10m
    if resolution == "10m":
        return fusionx2(X, models)
    elif resolution == "5m":
        return fusionx4(X, models)
    elif resolution == "2.5m":
        return fusionx8(X, models)
    else:
        raise ValueError("Invalid resolution. Please select 2.5m, 5m, or 10m.")


def fusionx2(
    X: torch.Tensor, 
    models: dict
) -> torch.Tensor:
    """Converts 20m bands to 10m resolution

    Args:
        X (torch.Tensor): The input tensor with the S2 bands
        models (dict): The dictionary with the loaded models

    Returns:
        torch.Tensor: The tensor with the same resolution for all the bands
    """

    # Obtain the device of X
    device = X.device

    # Band Selection
    index20 = [3, 4, 5, 7, 8, 9]
    index10 = [0, 1, 2, 6]

    # Set the model
    fusionmodelx2 = models["FusionX2"].to(device)

    # Select the 20m bands
    bands20_as_10 = X[index20]

    bands20 = torch.nn.functional.interpolate(
        bands20_as_10[None], scale_factor=0.5, mode="nearest"
    ).squeeze(0)

    bands20_in_10 = torch.nn.functional.interpolate(
        bands20[None], scale_factor=2, mode="bilinear", antialias=True
    ).squeeze(0)

    # Select the 10m bands
    bands10 = X[index10]

    # Concatenate the 20m and 10m bands
    input_data = torch.cat([bands20_in_10, bands10], dim=0)
    bands20_to_10 = fusionmodelx2(input_data[None]).squeeze(0)

    # Order the channels back
    results = torch.stack(
        [
            bands10[0],
            bands10[1],
            bands10[2],
            bands20_to_10[0],
            bands20_to_10[1],
            bands20_to_10[2],
            bands10[3],
            bands20_to_10[3],
            bands20_to_10[4],
            bands20_to_10[5],
        ],
        dim=0,
    )

    return results


def fusionx8(
    X: torch.Tensor, 
    models: dict
) -> torch.Tensor:
    """Converts 20m bands to 10m resolution

    Args:
        X (torch.Tensor): The input tensor with the S2 bands
        models (dict): The dictionary with the loaded models

    Returns:
        torch.Tensor: The tensor with the same resolution for all the bands
    """

    # Obtain the device of X
    device = X.device

    # Convert all bands to 10 meters
    superX: torch.Tensor = fusionx2(X, models)

    # Band Selection
    index20 = [3, 4, 5, 7, 8, 9]
    index10 = [2, 1, 0, 6]  # WARNING: The SR model needs RGBNIR bands

    # Set the SR resolution and x4 fusion model
    fusionmodelx4 = models["FusionX4"].to(device)
    srmodelx4 = models["SR"].to(device)

    # Convert the SWIR bands to 2.5m
    bands20_to_10 = superX[index20]
    bands10_in_2dot5 = torch.nn.functional.interpolate(
        bands20_to_10[None], scale_factor=4, mode="bilinear", antialias=True
    ).squeeze(0)

    # Run super-resolution on the 10m bands
    bands10 = superX[index10]
    bands10_to_2dot5 = srmodelx4(bands10[None]).squeeze(0)

    # Reorder the bands from RGBNIR to BGRNIR
    bands10_to_2dot5 = bands10_to_2dot5[[2, 1, 0, 3]]

    # Run the fusion x4 model in the SWIR bands (10m to 2.5m)
    input_data = torch.cat([bands10_in_2dot5, bands10_to_2dot5], dim=0)
    allbands_to_2dot5 = fusionmodelx4(input_data[None]).squeeze(0)

    # Order the channels back
    results = torch.stack(
        [
            bands10_to_2dot5[0],
            bands10_to_2dot5[1],
            bands10_to_2dot5[2],
            allbands_to_2dot5[0],
            allbands_to_2dot5[1],
            allbands_to_2dot5[2],
            bands10_to_2dot5[3],
            allbands_to_2dot5[3],
            allbands_to_2dot5[4],
            allbands_to_2dot5[5],
        ],
        dim=0,
    )

    return results


def fusionx4(
    X: torch.Tensor, 
    models: dict
) -> torch.Tensor:
    """Converts 20m bands to 10m resolution

    Args:
        X (torch.Tensor): The input tensor with the S2 bands
        models (dict): The dictionary with the loaded models

    Returns:
        torch.Tensor: The tensor with the same resolution for all the bands
    """

    # Obtain all the bands at 2.5m resolution
    superX = fusionx8(X, models)

    # From 2.5m to 5m resolution
    return torch.nn.functional.interpolate(
        superX[None], scale_factor=0.5, mode="bilinear", antialias=True
    ).squeeze(0)
