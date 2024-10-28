from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from supers2.xai.utils import gini, vis_saliency_kde


def GaussianBlurPath(sigma: float, fold: int, kernel_size: int = 5):
    """
    Generates a function for applying a Gaussian blur path to an image using PyTorch.
    The function applies progressively weaker Gaussian blurs to an image and calculates
    interpolations between each blurred image, along with derivatives for each step.

    Args:
        sigma (float): Initial standard deviation for the Gaussian blur.
        fold (int): Number of interpolation steps for the blurring path.
        kernel_size (int, optional): Size of the Gaussian kernel. Defaults to 5.

    Returns:
        Callable: A function that takes an image and returns a tuple:
            - image_interpolation (torch.Tensor): Interpolated blurred images.
            - lambda_derivative_interpolation (torch.Tensor): Derivatives of interpolated images.
    """

    def path_interpolation_func(torch_image: torch.Tensor):
        """
        Applies the Gaussian blur path to the input image and computes the interpolated images
        and their derivatives using PyTorch.

        Args:
            torch_image (torch.Tensor): Input image as a torch tensor (channels, height, width).

        Returns:
            tuple: Interpolated blurred images and their derivatives along the Gaussian path.
        """
        device = torch_image.device

        # Ensure image is 4D (batch, channels, height, width)
        torch_image = torch_image.unsqueeze(0) if torch_image.ndim == 3 else torch_image
        torch_image = torch_image.to(device)
        channels = torch_image.shape[1]

        # Initialize tensors for blurred images and derivatives
        image_interpolation = torch.zeros(
            (fold, *torch_image.shape[1:]), dtype=torch.float32
        )
        image_interpolation = image_interpolation.to(device)
        lambda_derivative_interpolation = torch.zeros_like(image_interpolation)
        lambda_derivative_interpolation = lambda_derivative_interpolation.to(device)
        kernel_interpolation = torch.zeros(
            (fold + 1, channels, kernel_size, kernel_size), dtype=torch.float32
        )
        kernel_interpolation = kernel_interpolation.to(device)

        # Linearly interpolate sigma values from initial to zero
        sigma_interpolation = np.linspace(sigma, 0, fold + 1)

        # Create Gaussian kernels for each sigma value
        for i in range(fold + 1):
            kernel_interpolation[i] = isotropic_gaussian_kernel_torch(
                kernel_size, sigma_interpolation[i]
            ).squeeze()

        # Calculate padding size
        pad_size = kernel_interpolation.shape[-1] // 2

        # Create Gaussian kernels for each sigma and apply to image
        for i in range(fold):
            # Apply reflect padding first
            padded_image = F.pad(
                torch_image, (pad_size, pad_size, pad_size, pad_size), mode="reflect"
            )

            # Store the current blurred image
            image_interpolation[i] = F.conv2d(
                padded_image, kernel_interpolation[i + 1][:, None], groups=channels
            ).squeeze(0)

            # Calculate derivative with respect to lambda
            diff_kernel = (kernel_interpolation[i + 1] - kernel_interpolation[i])[
                :, None
            ] * fold
            lambda_derivative_interpolation[i] = F.conv2d(
                padded_image, diff_kernel, groups=channels
            ).squeeze(0)

        return (
            image_interpolation,
            lambda_derivative_interpolation,
            kernel_interpolation,
        )

    return path_interpolation_func


def isotropic_gaussian_kernel_torch(
    size: int, sigma: float, epsilon: float = 1e-5
) -> torch.Tensor:
    """
    Generates an isotropic Gaussian kernel in PyTorch.

    Args:
        size (int): Size of the kernel (size x size).
        sigma (float): Standard deviation of the Gaussian distribution.
        epsilon (float, optional): Small constant to avoid division by zero. Defaults to 1e-5.

    Returns:
        torch.Tensor: Normalized Gaussian kernel.
    """
    ax = torch.arange(-size // 2 + 1.0, size // 2 + 1.0)
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")

    # Calculate Gaussian function and normalize
    kernel = torch.exp(-(xx**2 + yy**2) / (2.0 * (sigma + epsilon) ** 2))
    kernel = kernel / kernel.sum()

    # Reshape for 2D convolution (out_channels, in_channels, height, width)
    return kernel.unsqueeze(0).unsqueeze(0)


def Path_gradient(
    image: torch.Tensor,
    model: torch.nn.Module,
    attr_objective: callable,
    path_interpolation_func: callable,
):
    """
    Computes the path gradient for an image using a specified model and attribution objective.
    The function calculates gradients for a series of interpolated images produced by a path
    interpolation function.

    Args:
        numpy_image (np.ndarray): Input image of shape (channels, height, width).
        model (torch.nn.Module): The model to compute the objective on.
        attr_objective (callable): Function defining the attribution objective for the model output.
        path_interpolation_func (callable): Function that generates interpolated images and
            their lambda derivatives along a defined path.

    Returns:
        tuple:
            - grad_accumulate_list (np.ndarray): Accumulated gradients for each interpolated image.
            - results_numpy (np.ndarray): Model outputs for each interpolated image.
            - image_interpolation (np.ndarray): Interpolated images created by `path_interpolation_func`.
    """

    # Prepare image for interpolation and initialize gradient accumulation array
    image_interpolation, lambda_derivative_interpolation, _ = path_interpolation_func(
        image
    )

    grad_accumulate_list = torch.zeros_like(image_interpolation).cpu().numpy()
    result_list = []

    # Compute gradient for each interpolated image
    for i in range(image_interpolation.shape[0]):

        # Convert interpolated image to tensor and set requires_grad for backpropagation
        img_tensor = image_interpolation[i].float()[None]
        img_tensor.requires_grad_(True)

        # Forward pass through the model and compute attribution objective
        result = model(img_tensor)
        target = attr_objective(result)
        target.backward()  # Compute gradients

        # Extract gradient, handling NaNs if present
        grad = img_tensor.grad.cpu().numpy()
        grad = np.nan_to_num(grad)  # Replace NaNs with 0

        # Accumulate gradients adjusted by lambda derivatives
        grad_accumulate_list[i] = (
            grad * lambda_derivative_interpolation[i].cpu().numpy()
        )
        result_list.append(result.detach().cpu().numpy())

    # Collect results and return final outputs
    results_numpy = np.array(result_list)
    return grad_accumulate_list, results_numpy, image_interpolation


def attribution_objective(attr_func, h: int, w: int, window: int = 16):
    """
    Creates an objective function to calculate attribution within a specified window
    at given coordinates using an attribution function.

    Args:
        attr_func (Callable): A function that calculates attributions for an image.
        h (int): The top coordinate of the window within the image.
        w (int): The left coordinate of the window within the image.
        window (int, optional): The size of the square window. Defaults to 16.

    Returns:
        Callable: A function that takes an image as input and computes the attribution
        at the specified window location.
    """

    def calculate_objective(image):
        """
        Computes the attribution for a specified window within the given image.

        Args:
            image (torch.Tensor): A tensor representing the input image.

        Returns:
            torch.Tensor: The calculated attribution value within the specified window.
        """
        return attr_func(image, h, w, window=window)

    return calculate_objective


def attr_grad(
    tensor: torch.Tensor,
    h: int,
    w: int,
    window: int = 8,
    reduce: str = "sum",
    scale: float = 1.0,
) -> torch.Tensor:
    """
    Computes the gradient magnitude within a specified window of a 4D tensor and reduces the result.

    Args:
        tensor (torch.Tensor): A 4D tensor of shape (batch_size, channels, height, width).
        h (int): Starting height position of the window within the tensor.
        w (int): Starting width position of the window within the tensor.
        window (int, optional): The size of the square window to extract. Defaults to 8.
        reduce (str, optional): The reduction operation to apply to the window ('sum' or 'mean'). Defaults to 'sum'.
        scale (float, optional): Scaling factor to apply to the gradient magnitude. Defaults to 1.0.

    Returns:
        torch.Tensor: The reduced gradient magnitude for the specified window.
    """

    # Get tensor dimensions
    height = tensor.size(2)
    width = tensor.size(3)

    # Compute horizontal gradients by taking the difference between adjacent rows
    h_grad = torch.pow(tensor[:, :, : height - 1, :] - tensor[:, :, 1:, :], 2)

    # Compute vertical gradients by taking the difference between adjacent columns
    w_grad = torch.pow(tensor[:, :, :, : width - 1] - tensor[:, :, :, 1:], 2)

    # Calculate gradient magnitude by summing squares of gradients and taking the square root
    grad_magnitude = torch.sqrt(h_grad[:, :, :, :-1] + w_grad[:, :, :-1, :])

    # Crop the gradient magnitude tensor to the specified window
    windowed_grad = grad_magnitude[:, :, h : h + window, w : w + window]

    # Apply reduction (sum or mean) to the cropped window
    if reduce == "sum":
        return torch.sum(windowed_grad)
    elif reduce == "mean":
        return torch.mean(windowed_grad)
    else:
        raise ValueError(f"Invalid reduction type: {reduce}. Use 'sum' or 'mean'.")


def lam(
    X: torch.Tensor,
    model: torch.nn.Module,
    h: Optional[int] = 240,
    w: Optional[int] = 240,
    window: Optional[int] = 32,
    fold: Optional[int] = 25,
    kernel_size: Optional[int] = 13,
    sigma: Optional[float] = 3.5,
    robustness_metric: Optional[str] = True
):
    """
    Computes the Local Attribution Map (LAM) for an input tensor using 
    a specified model and attribution function. The function calculates
    the path gradient for each band in the input tensor and combines the
    results to generate the LAM.

    Args:
        X (torch.Tensor): Input tensor of shape (channels, height, width).
        model (torch.nn.Module): The model to compute the objective on.
        attr_func (callable): Function that calculates attributions for an image.
        h (int): The top coordinate of the window within the image.
        w (int): The left coordinate of the window within the image.
        window (int, optional): The size of the square window. Defaults to 16.
        fold (int, optional): Number of interpolation steps for the blurring path.
            Defaults to 10.
        kernel_size (int, optional): Size of the Gaussian kernel. Defaults to 5.
        sigma (float, optional): Initial standard deviation for the Gaussian blur.
            Defaults to 3.5.
        robustness_metric (bool, optional): Whether to return the robustness metric.
            Defaults to True.

    Returns:
        tuple: A tuple containing the following elements:
            - kde_map (np.ndarray): KDE estimation of the LAM.
            - complexity_metric (float): Gini index of the LAM that 
                measures the consistency of the attribution. The 
                larger the value, the more use more complex attribution
                patterns to solve the task.
            - robustness_metric (np.ndarray): Blurriness sensitivity of the LAM.
                The sensitivity measures the average gradient magnitude of the
                interpolated images.
            - robustness_vector (np.ndarray): Vector of gradient magnitudes for
                each interpolated image.
    """

    # Get the scale of the results
    with torch.no_grad():
        output = model(X[None])
        scale = output.shape[-1] // X.shape[-1]

    # Create the path interpolation function
    path_interpolation_func = GaussianBlurPath(
        sigma=sigma, fold=fold, kernel_size=kernel_size
    )
    # a, b, c = path_interpolation_func(X)

    # Create the attribution objective function
    attr_objective = attribution_objective(attr_grad, h, w, window=window)

    # Compute the path gradient for the input tensor
    grad_accumulate_list,results_numpy, image_interpolation = Path_gradient(
        X, model, attr_objective, path_interpolation_func
    )
    
    # Sum the accumulated gradients across all bands
    lam_results = torch.sum(torch.from_numpy(np.abs(grad_accumulate_list)), dim=0)
    grad_2d = np.abs(lam_results.sum(axis=0))
    grad_max = grad_2d.max()
    grad_norm = grad_2d / grad_max

    # Estimate gini index
    gini_index = gini(grad_norm.flatten())

    # KDE estimation
    kde_map = vis_saliency_kde(grad_norm, scale=scale, bandwidth=1.0)
    complexity_metric = (1 - gini_index) * 100

    # Estimate blurriness sensitivity
    robustness_vector = np.abs(grad_accumulate_list).mean(axis=(1, 2, 3))
    robustness_metric = np.trapz(robustness_vector)

    # Return the LAM results
    return kde_map, complexity_metric, robustness_metric, robustness_vector