import torch

def srmodel(
    sr_model: torch.nn.Module,
    f2_model: torch.nn.Module,
    reference_model_x4: torch.nn.Module,
    reference_model_hard_constraint_x4: torch.nn.Module,
    device: str = "cpu",
):

    reference_model_x4 = reference_model_x4.to(device)
    reference_model_hard_constraint_x4 = reference_model_hard_constraint_x4.to(device)
    reference_model_hard_constraint_x4.eval()
    for param in reference_model_hard_constraint_x4.parameters():
        param.requires_grad = False


    class SRModel(torch.nn.Module):
        def __init__(
            self,
            sr_model,
            f2_model,
            f4_model,
            f4_hard_constraint,
        ):
            super().__init__()
            self.sr_model = sr_model       
            self.f2_model = f2_model            
            self.f4_model = f4_model
            self.f4_hard_constraint = f4_hard_constraint            

        def forward(self, x):
            # Band Selection
            bands_20m = [3, 4, 5, 7, 8, 9]
            bands_10m = [2, 1, 0, 6]

            # Run Referece SR in the RSWIR bands (from 20m to 10m)
            allbands10m = self.f2_model(x)
                        
            # Convert the SWIR bands from 10m to 2.5m
            rsiwr_10m = allbands10m[:, bands_20m]
            rsiwr_2dot5m_billinear = torch.nn.functional.interpolate(
                rsiwr_10m, scale_factor=4, mode="bilinear", antialias=True
            )
            
            # Run SR in the RGBN bands (from 10m to 2.5m)
            rgbn_2dot5m = self.sr_model(x[:, bands_10m])

            # Reorder the bands from RGBNIR to BGRNIR
            rgbn_2dot5m = rgbn_2dot5m[:, [2, 1, 0, 3]]
            
            # Run the fusion x4 model in the SWIR bands (10m to 2.5m)
            input_data = torch.cat([rsiwr_2dot5m_billinear, rgbn_2dot5m], dim=1)
            rswirs2dot5 = self.f4_model(input_data)
            rswirs2dot5 = self.f4_hard_constraint(rsiwr_10m, rswirs2dot5)

            # Reconstruct sentinel-2 bands            
            return torch.stack([
                rgbn_2dot5m[:, 0],
                rgbn_2dot5m[:, 1],
                rgbn_2dot5m[:, 2],
                rswirs2dot5[:, 0],
                rswirs2dot5[:, 1],
                rswirs2dot5[:, 2],
                rgbn_2dot5m[:, 3],
                rswirs2dot5[:, 3],
                rswirs2dot5[:, 4],
                rswirs2dot5[:, 5],
            ], dim=1)
    
    return SRModel(sr_model, f2_model, reference_model_x4, reference_model_hard_constraint_x4)