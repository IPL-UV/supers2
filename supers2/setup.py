import importlib
import pathlib
from typing import Union

import torch

from supers2.dataclass import SRweights
from supers2.models.tricks import CNNHardConstraint
from supers2.trained_models import AllModels


class CustomModel(torch.nn.Module):
    """
    A custom model that applies a super-resolution model followed by a hard constraint.

    Attributes:
        sr_model (torch.nn.Module): The super-resolution model.
        hard_constraint (torch.nn.Module): The hard constraint applied after the super-resolution model.
    """

    def __init__(
        self, SRmodel: torch.nn.Module, HardConstraint: torch.nn.Module
    ) -> None:
        super(CustomModel, self).__init__()
        self.sr_model = SRmodel
        self.hard_constraint = HardConstraint

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that applies the super-resolution model and then the hard constraint.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying both SR model and hard constraint.
        """
        sr = self.sr_model(x)
        return self.hard_constraint(x, sr)


def load_model(
    snippet: str,
    weights_path: Union[str, pathlib.Path, None] = None,
    force_download: bool = False,
    device: str = "cpu",
) -> CustomModel:
    """
    Load a super-resolution model with pre-trained weights and apply a hard constraint.

    Args:
        snippet (str): Model identifier, case-insensitive.
        weights_path (Union[str, pathlib.Path, None]): Path to the model weights.
        force_download (bool): If True, downloads the model weights if not available.

    Returns:
        CustomModel: A model instance with both SR and hard constraint applied.
    """

    # Normalize snippet to lowercase for case-insensitive matching
    snippet = snippet.lower()

    # Is a zero-parameter model?
    if "__simple__" not in snippet:
        # Retrieve model weights information and validate snippet and path
        model_weights = SRweights(
            snippet=snippet, path=weights_path, force_download=force_download
        )
        model_fullpath = model_weights.fullname

        # Load the weights
        model_weights_data = torch.load(
            model_fullpath, map_location=torch.device("cpu"), weights_only=True
        )

    # Dynamically load the model class based on the specified snippet
    modelclass_path = AllModels.object[snippet].srclass
    modelmodule, modelclass_name = modelclass_path.rsplit(".", 1)
    modelclass_module = importlib.import_module(modelmodule)
    modelclass = getattr(modelclass_module, modelclass_name)

    # Initialize the model with its specified parameters
    model_parameters = AllModels.object[snippet].parameters
    model_parameters["device"] = device
    model = modelclass(**model_parameters)
    model.load_state_dict(model_weights_data) if "__simple__" not in snippet else None
    model.eval()  # Set model to evaluation mode
    model.to(device)  # Move model to device
    for param in model.parameters():
        param.requires_grad = False  # Freeze model parameters

    # Define the Hard Constraint
    # upscale = 1 [FUSION]; upscale = 4 [SR]; upscale = 2 [FusionX2]
    if model_parameters["upscale"] == 1:
        hard_constraint_parameters = {
            "filter_method": "butterworth",
            "filter_hyperparameters": {"order": 6},
            "scale_factor": model_parameters["upscale"] * 2,
            "in_channels": 6,
            "out_channels": [0, 1, 2, 3, 4, 5],
        }
    else:
        hard_constraint_parameters = {
            "filter_method": "butterworth",
            "filter_hyperparameters": {"order": 6},
            "scale_factor": model_parameters["upscale"] * 2,
            "in_channels": 4,
            "out_channels": [0, 1, 2, 3],
        }

    # Instantiate the Hard Constraint
    hard_constraint = CNNHardConstraint(**hard_constraint_parameters)
    hard_constraint.eval()
    hard_constraint.to(device)
    for param in hard_constraint.parameters():
        param.requires_grad = False

    # Return the combined model with the hard constraint applied
    return CustomModel(SRmodel=model, HardConstraint=hard_constraint)
