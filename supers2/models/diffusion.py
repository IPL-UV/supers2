import torch
from supers2.models.utils import assert_tensor_validity
from supers2.models.utils import revert_padding
from supers2.models.opensr_diffusion.diffusion.utils import DDIMSampler
from supers2.models.opensr_diffusion.diffusion.latentdiffusion import LatentDiffusion

from skimage.exposure import match_histograms

from tqdm import tqdm
import numpy as np
from opensr_model.utils import linear_transform_4b


class SRLatentDiffusion(torch.nn.Module):
    def __init__(self, device: str = "cpu"):
        super().__init__()

        # Set up the model
        first_stage_config, cond_stage_config = self.set_model_settings()
        self.model = LatentDiffusion(
            first_stage_config,
            cond_stage_config,
            timesteps=1000,
            unet_config=cond_stage_config,
            linear_start=0.0015,
            linear_end=0.0155,
            concat_mode=True,
            cond_stage_trainable=False,
            first_stage_key="image",
            cond_stage_key="LR_image",
        )
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False
        
        # Set up the model for inference
        self.device = device # set self device
        self.model.device = device # set model device as selected
        self.model = self.model.to(device) # move model to device
        self.model.eval() # set model state
        self._X = None # placeholder for LR image
        self.encode_conditioning = True # encode LR images before dif?
                

    def set_model_settings(self):
        # set up model settings
        first_stage_config = {
            "embed_dim":4,
            "double_z": True,
            "z_channels": 4,
            "resolution": 256,
            "in_channels": 4,
            "out_ch": 4,
            "ch": 128,
            "ch_mult": [1, 2, 4],
            "num_res_blocks": 2,
            "attn_resolutions": [],
            "dropout": 0.0,
        }
        cond_stage_config = {
            "image_size": 64,
            "in_channels": 8,
            "model_channels": 160,
            "out_channels": 4,
            "num_res_blocks": 2,
            "attention_resolutions": [16, 8],
            "channel_mult": [1, 2, 2, 4],
            "num_head_channels": 32,
        }                        
        self.linear_transform = linear_transform_4b

        return first_stage_config, cond_stage_config

        
    def _tensor_encode(self,X: torch.Tensor):
        # set copy to model
        #X = torch.rand(1, 4, 32, 32)
        self._X = X.clone()
        # normalize image    
        X_enc = self.linear_transform(X, stage="norm")
        
        # encode LR images
        X_int = torch.nn.functional.interpolate(X, size=X.shape[-1]*4, mode='bilinear', antialias=True)
        
        # encode conditioning
        X_enc = self.model.first_stage_model.encode(X_int).sample()
                
        return X_enc

    def _tensor_decode(self, X_enc: torch.Tensor):
        
        # Decode
        X_dec = self.model.decode_first_stage(X_enc)
        X_dec = self.linear_transform(X_dec, stage="denorm")
        
        # Apply spectral correction
        for i in range(X_dec.shape[1]):
            X_dec[:, i] = self.hq_histogram_matching(X_dec[:, i], self._X[:, i])

        # If the value is negative, set it to 0
        X_dec[X_dec < 0] = 0
        
        return X_dec
    
    def _prepare_model(
        self,
        X: torch.Tensor,
        eta: float = 1.0,
        custom_steps: int = 100,
        verbose: bool = False 
    ):
        # Create the DDIM sampler
        ddim = DDIMSampler(self.model)
        
        # make schedule to compute alphas and sigmas
        ddim.make_schedule(ddim_num_steps=custom_steps, ddim_eta=eta, verbose=verbose)
        
        # Create the HR latent image
        latent = torch.randn(X.shape, device=X.device)
                
        # Create the vector with the timesteps
        timesteps = ddim.ddim_timesteps
        time_range = np.flip(timesteps)
        
        return ddim, latent, time_range

    @torch.no_grad()
    def forward(
        self,
        X: torch.Tensor,
        eta: float = 1.0,
        custom_steps: int = 100,
        temperature: float = 1.0,
        verbose: bool = True
    ):
        """Obtain the super resolution of the given image.

        Args:
            X (torch.Tensor): If a Sentinel-2 L2A image with reflectance values
                in the range [0, 1] and shape CxWxH, the super resolution of the image
                is returned. If a batch of images with shape BxCxWxH is given, a batch
                of super resolved images is returned.
            custom_steps (int, optional): Number of steps to run the denoiser. Defaults
                to 100.
            temperature (float, optional): Temperature to use in the denoiser.
                Defaults to 1.0. The higher the temperature, the more stochastic
                the denoiser is (random noise gets multiplied by this).
            spectral_correction (bool, optional): Apply spectral correction to the SR
                image, using the LR image as reference. Defaults to True.

        Returns:
            torch.Tensor: The super resolved image or batch of images with a shape of
                Cx(Wx4)x(Hx4) or BxCx(Wx4)x(Hx4).
        """
        
        # Assert shape, size, dimensionality
        X, padding = assert_tensor_validity(X)

        # Normalize the image
        X = X.clone()        
        Xnorm = self._tensor_encode(X)
        
        # ddim, latent and time_range
        ddim, latent, time_range = self._prepare_model(
            X=Xnorm, eta=eta, custom_steps=custom_steps, verbose=False
        )
        iterator = tqdm(time_range, desc="DDIM Sampler", total=custom_steps,disable=not verbose)

        # Iterate over the timesteps
        for i, step in enumerate(iterator):
            outs = ddim.p_sample_ddim(
                x=latent,
                c=Xnorm,
                t=step,
                index=custom_steps - i - 1,
                use_original_steps=False,
                temperature=temperature
            )
            latent, _ = outs
        
        sr = self._tensor_decode(latent)
        sr = revert_padding(sr,padding)
        return sr


    def hq_histogram_matching(
        self, image1: torch.Tensor, image2: torch.Tensor
    ) -> torch.Tensor:
        """Lazy implementation of histogram matching

        Args:
            image1 (torch.Tensor): The low-resolution image (C, H, W).
            image2 (torch.Tensor): The super-resolved image (C, H, W).

        Returns:
            torch.Tensor: The super-resolved image with the histogram of
                the target image.
        """

        # Go to numpy
        np_image1 = image1.detach().cpu().numpy()
        np_image2 = image2.detach().cpu().numpy()

        if np_image1.ndim == 3:
            np_image1_hat = match_histograms(np_image1, np_image2, channel_axis=0)
        elif np_image1.ndim == 2:
            np_image1_hat = match_histograms(np_image1, np_image2, channel_axis=None)
        else:
            raise ValueError("The input image must have 2 or 3 dimensions.")

        # Go back to torch
        image1_hat = torch.from_numpy(np_image1_hat).to(image1.device)

        return image1_hat
    