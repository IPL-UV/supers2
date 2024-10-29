import torch


class BilinearSR(torch.nn.Module):
    """
    A simple super-resolution model that uses bilinear interpolation.

    Attributes:
        scale_factor (int): The upscaling factor.
    """

    def __init__(self, device: str = "cpu", scale_factor: int = 4, **kwargs) -> None:
        super(BilinearSR, self).__init__()
        self.scale_factor = scale_factor
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that applies bilinear interpolation.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying bilinear interpolation.
        """
        return torch.nn.functional.interpolate(
            x,
            scale_factor=self.scale_factor,
            mode="bilinear",
            antialias=True,
        )
    

class BicubicSR(torch.nn.Module):
    """   A simple super-resolution model that uses bicubic interpolation.

    Attributes:
        scale_factor (int): The upscaling factor.
    """

    def __init__(self, device: str = "cpu", scale_factor: int = 4, **kwargs) -> None:
        super(BicubicSR, self).__init__()
        self.scale_factor = scale_factor
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that applies bicubic interpolation.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying bicubic interpolation.
        """
        return torch.nn.functional.interpolate(
            x,
            scale_factor=self.scale_factor,
            mode="bicubic",
            antialias=True,
        )