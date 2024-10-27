import pathlib
from typing import Literal, Optional, Union

import torch

from supers2.dataclass import SRexperiment
from supers2.setup import load_model


def setmodel(
    resolution: Literal["2.5m", "5m", "10m"] = "2.5m",
    sr_model_snippet: str = "sr__opensrbaseline__cnn__lightweight__l1",
    fusionx2_model_snippet: str = "fusionx2__opensrbaseline__cnn__lightweight__l1",
    fusionx4_model_snippet: str = "fusionx4__opensrbaseline__cnn__lightweight__l1",
    weights_path: Union[str, pathlib.Path, None] = None,
    device: str = "cpu",
    **kwargs
) -> SRexperiment:
    
    # For experiments that only require 10m resolution
    if resolution == "10m":
        return SRexperiment(
            Fusionx2=load_model(
                snippet=fusionx2_model_snippet, weights_path=weights_path, device=device
            ),
            Fusionx4=None,
            SRx4=None,
        )

    return SRexperiment(
        fusionx2=load_model(
            snippet=fusionx2_model_snippet, weights_path=weights_path, device=device
        ),
        fusionx4=load_model(
            snippet=fusionx4_model_snippet, weights_path=weights_path, device=device
        ),
        srx4=load_model(
            snippet=sr_model_snippet, weights_path=weights_path, device=device, **kwargs
        ),
    )


def predict(
    X: torch.Tensor,
    resolution: Literal["2.5m", "5m", "10m"] = "2.5m",
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


def fusionx2(X: torch.Tensor, models: dict) -> torch.Tensor:
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
    bands_20m = [3, 4, 5, 7, 8, 9]
    bands_10m = [0, 1, 2, 6]

    # Set the model
    fusionmodelx2 = models.fusionx2.to(device)

    # Select the 20m bands
    bands_20m_data = X[bands_20m]

    bands_20m_data_real = torch.nn.functional.interpolate(
        bands_20m_data[None], scale_factor=0.5, mode="nearest"
    ).squeeze(0)

    bands_20m_data = torch.nn.functional.interpolate(
        bands_20m_data_real[None], scale_factor=2, mode="bilinear", antialias=True
    ).squeeze(0)

    # Select the 10m bands
    bands_10m_data = X[bands_10m]

    # Concatenate the 20m and 10m bands
    input_data = torch.cat([bands_20m_data, bands_10m_data], dim=0)
    bands_20m_data_to_10 = fusionmodelx2(input_data[None]).squeeze(0)

    # Order the channels back
    results = torch.stack(
        [
            bands_10m_data[0],
            bands_10m_data[1],
            bands_10m_data[2],
            bands_20m_data_to_10[0],
            bands_20m_data_to_10[1],
            bands_20m_data_to_10[2],
            bands_10m_data[3],
            bands_20m_data_to_10[3],
            bands_20m_data_to_10[4],
            bands_20m_data_to_10[5],
        ],
        dim=0,
    )

    return results


def fusionx8(X: torch.Tensor, models: dict) -> torch.Tensor:
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
    bands_20m = [3, 4, 5, 7, 8, 9]
    bands_10m = [2, 1, 0, 6]  # WARNING: The SR model needs RGBNIR bands

    # Set the SR resolution and x4 fusion model
    fusionmodelx4 = models.fusionx4.to(device)
    srmodelx4 = models.srx4.to(device)

    # Convert the SWIR bands to 2.5m
    bands_20m_data = superX[bands_20m]
    bands_20m_data_up = torch.nn.functional.interpolate(
        bands_20m_data[None], scale_factor=4, mode="bilinear", antialias=True
    ).squeeze(0)

    # Run super-resolution on the 10m bands
    rgbn_bands_10m_data = superX[bands_10m]
    tensor_x4_rgbnir = srmodelx4(rgbn_bands_10m_data[None]).squeeze(0)

    # Reorder the bands from RGBNIR to BGRNIR
    tensor_x4_rgbnir = tensor_x4_rgbnir[[2, 1, 0, 3]]

    # Run the fusion x4 model in the SWIR bands (10m to 2.5m)
    input_data = torch.cat([bands_20m_data_up, tensor_x4_rgbnir], dim=0)
    bands_20m_data_to_25m = fusionmodelx4(input_data[None]).squeeze(0)

    # Order the channels back
    results = torch.stack(
        [
            tensor_x4_rgbnir[0],
            tensor_x4_rgbnir[1],
            tensor_x4_rgbnir[2],
            bands_20m_data_to_25m[0],
            bands_20m_data_to_25m[1],
            bands_20m_data_to_25m[2],
            tensor_x4_rgbnir[3],
            bands_20m_data_to_25m[3],
            bands_20m_data_to_25m[4],
            bands_20m_data_to_25m[5],
        ],
        dim=0,
    )

    return results


def fusionx4(X: torch.Tensor, models: dict) -> torch.Tensor:
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