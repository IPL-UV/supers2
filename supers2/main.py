import pathlib
from typing import Literal, Optional, Union, Tuple

import numpy as np
import rasterio as rio
import torch
import tqdm

from supers2.dataclass import SRexperiment
from supers2.setup import load_model
from supers2.utils import define_iteration, gdal_create
from supers2.trained_models import SRmodels


def setmodel(
    resolution: Literal["2.5m", "5m", "10m"] = "2.5m",
    sr_model_snippet: str = "sr__opensrbaseline__cnn__lightweight__l1",
    fusionx2_model_snippet: str = "fusionx2__opensrbaseline__cnn__lightweight__l1",
    fusionx4_model_snippet: str = "fusionx4__opensrbaseline__cnn__lightweight__l1",
    weights_path: Union[str, pathlib.Path, None] = None,
    device: str = "cpu",
    **kwargs,
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
        models = setmodel(resolution=resolution, device=X.device)

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


def predict_large(
    image_fullname: Union[str, pathlib.Path],
    output_fullname: Union[str, pathlib.Path],
    resolution: Literal["2.5m", "5m", "10m"] = "2.5m",
    overlap: int = 32,
    models: Optional[dict] = None,
    device: str = "cpu",
) -> pathlib.Path:
    """Generate a new S2 tensor with all the bands on the same resolution

    Args:
        image_fullname (Union[str, pathlib.Path]): The input image with the S2 bands
        output_fullname (Union[str, pathlib.Path]): The output image with the S2 bands
        resolution (Literal["2.5m", "5m", "10m"], optional): The final resolution of the
            tensor. Defaults to "2.5m".
        models (Optional[dict], optional): The dictionary with the loaded models. Defaults
            to None.

    Returns:
        pathlib.Path: The path to the output image
    """

    # Define the resolution factor
    if resolution == "2.5m":
        res_n = 4
    elif resolution == "5m":
        res_n = 2
    elif resolution == "10m":
        res_n = 1
    else:
        raise ValueError("The resolution is not valid")

    # Get the image metadata and check if the image is tiled
    with rio.open(image_fullname) as src:
        metadata = src.profile
        if metadata["tiled"] == False:
            raise ValueError("The image is not tiled")
        if metadata["blockxsize"] != 128 or metadata["blockysize"] != 128:
            raise ValueError("The image does not have 128x128 blocks")

    # Run always in patches of 128x128 with 32 of overlap
    nruns = define_iteration(
        dimension=(metadata["height"], metadata["width"]),
        chunk_size=128,
        overlap=overlap,
    )

    # Define the output metadata
    output_metadata = metadata.copy()
    output_metadata["width"] = metadata["width"] * res_n
    output_metadata["height"] = metadata["height"] * res_n
    output_metadata["transform"] = rio.transform.Affine(
        metadata["transform"].a / res_n,
        metadata["transform"].b,
        metadata["transform"].c,
        metadata["transform"].d,
        metadata["transform"].e / res_n,
        metadata["transform"].f,
    )
    output_metadata["blockxsize"] = 128 * res_n
    output_metadata["blockysize"] = 128 * res_n

    # Create the output image
    with rio.open(output_fullname, "w", **output_metadata) as dst:
        pass

    # Check if the models are loaded
    if models is None:
        models = setmodel(resolution=resolution, device=device)

    # Iterate over the image
    with rio.open(output_fullname, "r+") as dst:
        with rio.open(image_fullname) as src:
            for index, point in enumerate(tqdm.tqdm(nruns)):
                # Read a block of the image
                window = rio.windows.Window(point[1], point[0], 128, 128)
                X = torch.from_numpy(src.read(window=window)).float().to(device)

                # Predict the super-resolution
                result = (
                    predict(X=X / 10_000, models=models, resolution=resolution) * 10_000
                )
                result[result < 0] = 0
                result = result.cpu().numpy().astype(np.uint16)

                # Define the offset in the output space
                # If the point is at the border, the offset is 0
                # otherwise consider the overlap
                if point[1] == 0:
                    offset_x = 0
                else:
                    offset_x = point[1] * res_n + overlap * res_n // 2

                if point[0] == 0:
                    offset_y = 0
                else:
                    offset_y = point[0] * res_n + overlap * res_n // 2

                # Define the length of the patch
                # The patch is always 224x224
                # There is three conditions:
                #  - The patch is at the corner begining (0, *) or (*, 0)
                #  - The patch is at the corner ending (width, *) or (*, height)
                #  - The patch is in the middle of the image
                if offset_x == 0:
                    skip = overlap * res_n // 2
                    length_x = 128 * res_n - skip
                    result = result[:, :, :length_x]
                elif (offset_x + 128) == metadata["width"]:
                    length_x = 128 * res_n
                    result = result[:, :, :length_x]
                else:
                    skip = overlap * res_n // 2
                    length_x = 128 * res_n - skip
                    result = result[:, :, skip : (128 * res_n)]

                # Do the same for the Y axis
                if offset_y == 0:
                    skip = overlap * res_n // 2
                    length_y = 128 * res_n - skip
                    result = result[:, :length_y, :]
                elif (offset_y + 128) == metadata["height"]:
                    length_y = 128 * res_n
                    result = result[:, :length_y, :]
                else:
                    skip = overlap * res_n // 2
                    length_y = 128 * res_n - overlap * res_n // 2
                    result = result[:, skip : (128 * res_n), :]

                # Write the result in the output image
                window = rio.windows.Window(offset_x, offset_y, length_x, length_y)
                dst.write(result, window=window)

    return pathlib.Path(output_fullname)


def predict_rgbnir(
    X: torch.Tensor,
    resolution: Literal["2.5m", "5m"] = "2.5m",
    sr_model_snippet: Optional[str] = "sr__opensrbaseline__cnn__lightweight__l1",
    weights_path: Optional[Union[str, pathlib.Path]] = None,
    device: str = "cpu",
    **kwargs,
) -> torch.Tensor:
    """Generate a new S2 tensor with RGBNIR bands on the same resolution

    Args:
        X (torch.Tensor): The input tensor with the S2 bands (RGBNIR)
        device (str, optional): The device to use. Defaults to "cpu".

    Returns:
        torch.Tensor: The tensor with the same resolution for all the bands
    """
    # Device of the input tensor
    device = X.device

    # Check if the models are loaded
    model = load_model(
        snippet=sr_model_snippet, weights_path=weights_path, device=device, **kwargs
    )
    model = model.to(device)

    # Run the super-resolution
    result = model(X[None]).squeeze(0)

    if resolution == "5m":
        result = torch.nn.functional.interpolate(
            result[None], scale_factor=0.5, mode="bilinear", antialias=True
        ).squeeze(0)

    return result


def uncertainty(
    X: torch.Tensor,
    models: str = "all",
    weights_path: str = None,
    device: str = "cpu",
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate the mean and standard deviation of the super-resolution models

    Args:
        X (torch.Tensor): The input tensor with the S2 bands (RGBNIR)
        models (str, optional): The models to use. Defaults to "all".
        weights_path (str, optional): The path to the weights. Defaults
            to None.
        device (str, optional): The device to use. Defaults to "cpu".

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The mean and standard deviation
    """

    if models == "all":
        models = list(SRmodels.model_dump()["object"].keys())

    container = []
    for model in tqdm.tqdm(models):
        # Load a model
        model_object = load_model(
            snippet=model, weights_path=weights_path, device=device, **kwargs
        )

        # Run the model
        X_torch = X.float().to(device)
        prediction = model_object(X_torch[None]).squeeze().cpu()

        # Store the prediction
        container.append(prediction)

    # Calculate the mean and standard deviation
    mean = torch.stack(container).mean(dim=0)
    std = torch.stack(container).std(dim=0)

    return mean, std
