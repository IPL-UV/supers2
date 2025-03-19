import torch

def resample_sentinel2_bands(X):
    # S2 bands indices
    indices_20m = [3, 4, 5, 7, 8, 9]
    indices_10m = [0, 1, 2, 6]
    
    # Extract bands
    bandas_20m = X[:, indices_20m]
    bandas_10m = X[:, indices_10m]
    
    # First upsample with nearest interpolation to 20m
    real_b20m = torch.nn.functional.interpolate(
        bandas_20m, scale_factor=0.5, mode="nearest"
    )
    
    # Then upsample with bilinear interpolation to 10m
    smooth_b20m = torch.nn.functional.interpolate(
        real_b20m, scale_factor=2, mode="bilinear", antialias=True
    )
    
    # Concatenate bands
    return torch.cat([smooth_b20m, bandas_10m], dim=1)


def reconstruct_sentinel2_stack(b10m, b20m):

    return torch.stack(
        [
            b10m[:, 0], # B2 (Blue)
            b10m[:, 1], # B3 (Green)
            b10m[:, 2], # B4 (Red)
            b20m[:, 0], # B5 (Red Edge 1)
            b20m[:, 1], # B6 (Red Edge 2)
            b20m[:, 2], # B7 (Red Edge 3)
            b10m[:, 3], # B8 (NIR)
            b20m[:, 3], # B8A (Narrow NIR)
            b20m[:, 4], # B11 (SWIR 1)
            b20m[:, 5], # B12 (SWIR 2)
        ],
        dim=1,
    )



def srmodel(
    sr_model: torch.nn.Module,
    hard_constraint: torch.nn.Module,
    device: str = "cpu",
):
    # Load the SR model    
    sr_model.to(device)

    # Load HardConstraint
    hard_constraint = hard_constraint.eval()
    for param in hard_constraint.parameters():
        param.requires_grad = False
    hard_constraint = hard_constraint.to(device)

    class SRModel(torch.nn.Module):
        def __init__(self, sr_model, hard_constraint):
            super().__init__()
            self.sr_model = sr_model
            self.hard_constraint = hard_constraint

        def forward(self, x):
            # Resample sentinel-2 bands order
            rgbn = x[:, [0, 1, 2, 6]].clone()
            x = resample_sentinel2_bands(x)

            # Run model
            sr = self.sr_model(x)
            sr = self.hard_constraint(x, sr)

            # Reconstruct sentinel-2 bands
            sr = reconstruct_sentinel2_stack(rgbn, sr)

            return sr
            
    return SRModel(sr_model, hard_constraint)


