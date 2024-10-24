from typing import Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def ideal_filter(
    shape: Tuple[int, int], 
    cutoff: int
) -> torch.Tensor:
    """
    Creates an ideal low-pass filter.

    Args:
        shape: (rows, cols) of the filter.
        cutoff: Cutoff radius for the filter.

    Returns:
        torch.Tensor: Normalized ideal filter.
    """


def butterworth_filter(
    shape: Tuple[int, int], 
    cutoff: int, 
    order: int
) -> torch.Tensor:
    """
    Creates a Butterworth low-pass filter.

    Args:
        shape: (rows, cols) of the filter.
        cutoff: Cutoff frequency.
        order: Order of the Butterworth filter.

    Returns:
        torch.Tensor: Normalized Butterworth filter.
    """
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    filter = torch.zeros((rows, cols), dtype=torch.float32)
    for u in range(rows):
        for v in range(cols):
            distance = ((u - crow) ** 2 + (v - ccol) ** 2) ** 0.5
            filter[u, v] = 1 / (1 + (distance / cutoff) ** (2 * order))
    filter /= filter.sum()
    return filter


def gaussian_filter(
    shape: Tuple[int, int], 
    cutoff: int
) -> torch.Tensor:
    """
    Creates a Gaussian low-pass filter.

    Args:
        shape: (rows, cols) of the filter.
        cutoff: Standard deviation for the Gaussian filter.

    Returns:
        torch.Tensor: Normalized Gaussian filter.
    """
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    filter = torch.zeros((rows, cols), dtype=torch.float32)
    for u in range(rows):
        for v in range(cols):
            distance = (u - crow) ** 2 + (v - ccol) ** 2
            filter[u, v] = torch.exp(-distance / (2 * (cutoff**2)))
    filter /= filter.sum()
    return filter


def sigmoid_filter(
    shape: Tuple[int, int], 
    cutoff: int, 
    sharpness: float
) -> torch.Tensor:
    """
    Creates a Sigmoid-based low-pass filter.

    Args:
        shape: (rows, cols) of the filter.
        cutoff: Cutoff frequency.
        sharpness: Sharpness of the transition in the filter.

    Returns:
        torch.Tensor: Normalized Sigmoid filter.
    """
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    filter = torch.zeros((rows, cols), dtype=torch.float32)
    for u in range(rows):
        for v in range(cols):
            distance = ((u - crow) ** 2 + (v - ccol) ** 2) ** 0.5
            filter[u, v] = 1 / (1 + torch.exp((distance - cutoff) / sharpness))
    filter /= filter.sum()
    return filter


class FourierHardConstraint(torch.nn.Module):
    """
    Applies a low-pass Fourier constraint using different filter methods.

    Args:
        filter_method: Filter type ('ideal', 'butterworth', 'sigmoid', 'gaussian').
        filter_hyperparameters: Hyperparameters for the chosen filter method.
        sr_image_size: Size of the super-resolution image (height, width).
        scale_factor: Scale factor for the super-resolution task.
        device: Device where the filters and tensors are located. Default is "cpu".
    """

    def __init__(
        self,
        filter_method: Literal["ideal", "butterworth", "sigmoid", "gaussian"],
        filter_hyperparameters: dict,
        sr_image_size: tuple,
        scale_factor: int,
        device: str = "cpu",
    ):
        super().__init__()
        h, w = sr_image_size
        center_h, center_w = h // 2, w // 2

        # Calculate the radius for the low-pass filter based on the scale factor
        if filter_method == "ideal":
            radius = min(center_h, center_w) // scale_factor
            low_pass_mask = ideal_filter((h, w), radius)
        elif filter_method == "butterworth":
            radius = min(center_h, center_w) // scale_factor
            low_pass_mask = butterworth_filter(
                (h, w), radius, order=filter_hyperparameters["order"]
            )
        elif filter_method == "gaussian":
            radius = min(center_h, center_w) // scale_factor
            low_pass_mask = gaussian_filter((h, w), radius)
        elif filter_method == "sigmoid":
            radius = min(center_h, center_w) // scale_factor
            low_pass_mask = sigmoid_filter(
                (h, w), radius, sharpness=filter_hyperparameters["sharpness"]
            )
        else:
            raise ValueError(f"Unsupported fourier_method: {filter_method}")

        self.low_pass_mask = low_pass_mask.to(device)
        self.scale_factor = scale_factor

    def forward(
        self, 
        lr: torch.Tensor, 
        sr: torch.Tensor
    ) -> torch.Tensor:
        """
        Applies the Fourier constraint on the super-resolution image.

        Args:
            lr: Low-resolution input tensor.
            sr: Super-resolution output tensor.

        Returns:
            torch.Tensor: Hybrid image after applying Fourier constraint.
        """
        # Upsample the LR image to the HR size
        lr_up = torch.nn.functional.interpolate(
            lr, size=sr.shape[-2:], mode="bicubic", antialias=True
        )

        # Apply the low-pass filter to the HR image
        sr_fft = torch.fft.fftn(sr, dim=(-2, -1))
        lr_fft = torch.fft.fftn(lr_up, dim=(-2, -1))

        # Shift the zero-frequency component to the center
        sr_fft_shifted = torch.fft.fftshift(sr_fft)
        lr_fft_shifted = torch.fft.fftshift(lr_fft)

        # High-pass filter is the complement of the low-pass filter
        high_pass_mask = 1 - self.low_pass_mask

        # Apply the high-pass filter to the SR image
        f1_low = lr_fft_shifted * self.low_pass_mask
        f1_high = sr_fft_shifted * high_pass_mask

        # Combine the low-pass and high-pass components
        sr_fft_filtered = f1_low + f1_high

        # Inverse FFT to get the filtered SR image
        combined_ishift = torch.fft.ifftshift(sr_fft_filtered)
        hybrid_image = torch.real(torch.fft.ifft2(combined_ishift))

        return hybrid_image


class CNNHardConstraint(nn.Module):
    """
    Applies a convolutional hard constraint using predefined filters for low-pass and high-pass filtering.

    Args:
        filter_method: The type of filter to apply ('ideal', 'butterworth', 'gaussian', 'sigmoid').
        filter_hyperparameters: Dictionary containing hyperparameters specific to the chosen filter method.
        scale_factor: Scaling factor used to determine kernel size and cutoff frequency.
        in_channels: Number of input channels.
        out_channels: List of channels to be processed (default is [0, 1, 2, 3, 4, 5]).
    """

    def __init__(
        self,
        filter_method: Literal["ideal", "butterworth", "gaussian", "sigmoid"],
        filter_hyperparameters: dict,
        scale_factor: int,
        in_channels: int,
        out_channels: list = [0, 1, 2, 3, 4, 5],
    ):
        super().__init__()

        self.filter_method = filter_method
        self.in_channels = in_channels
        self.filter_hyperparameters = filter_hyperparameters or {}

        # Estimate the kernel according to the scale
        kernel_size = scale_factor * 3 + 1
        cutoff = scale_factor * 2

        # Define the convolution layer with multiple input and output channels
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=len(out_channels),
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
            groups=in_channels,
        )

        # Remove the gradient for the filter weights
        self.conv.weight.requires_grad = False

        # Initialize the filter kernel based on the filter method
        if filter_method == "ideal":
            weight_data = ideal_filter((kernel_size, kernel_size), cutoff)
        elif filter_method == "butterworth":
            order = filter_hyperparameters.get("order", 2)
            weight_data = butterworth_filter((kernel_size, kernel_size), cutoff, order)
        elif filter_method == "gaussian":
            weight_data = gaussian_filter((kernel_size, kernel_size), cutoff)
        elif filter_method == "sigmoid":
            sharpness = filter_hyperparameters.get("sharpness", 1.0)
            weight_data = sigmoid_filter((kernel_size, kernel_size), cutoff, sharpness)
        else:
            raise ValueError(f"Unsupported filter_method: {filter_method}")

        # Apply the same filter to all input channels
        self.conv.weight.data = (
            weight_data.unsqueeze(0).unsqueeze(0).repeat(in_channels, 1, 1, 1)
        )
        self.out_channels = out_channels

    def forward(
        self, 
        lr: torch.Tensor, 
        sr: torch.Tensor
    ) -> torch.Tensor:
        """
        Applies the filter constraint on the super-resolution image.

        Args:
            lr: Low-resolution input tensor.
            sr: Super-resolution output tensor.

        Returns:
            torch.Tensor: The resulting hybrid image after applying the constraint.
        """
        # Upsample the LR image to the size of SR
        lr = lr[:, self.out_channels]

        # Upsample the LR image to the size of SR
        lr_up = F.interpolate(lr, size=sr.shape[-2:], mode="bicubic", antialias=True)

        # Apply the convolutional filter to both LR and SR images
        lr_filtered = self.conv(lr_up)
        sr_filtered = self.conv(sr)

        # Combine low-pass and high-pass components
        hybrid_image = lr_filtered + (sr - sr_filtered)

        return hybrid_image