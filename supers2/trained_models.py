from supers2.dataclass import AvailableModel, AvailableModels

SRmodels = AvailableModels(
    object={
        # CNN Models
        "sr__opensrbaseline__cnn__lightweight__l1": AvailableModel(
            parameters={
                "in_channels": 4,
                "out_channels": 4,
                "feature_channels": 24,
                "upscale": 4,
                "bias": True,
                "train_mode": False,
                "num_blocks": 6,
            },
            srclass="supers2.models.opensr_baseline.cnn.CNNSR",
        ),
        "sr__opensrbaseline__cnn__small__l1": AvailableModel(
            parameters={
                "in_channels": 4,
                "out_channels": 4,
                "feature_channels": 48,
                "upscale": 4,
                "bias": True,
                "train_mode": False,
                "num_blocks": 16,
            },
            srclass="supers2.models.opensr_baseline.cnn.CNNSR",
        ),
        "sr__opensrbaseline__cnn__medium__l1": AvailableModel(
            parameters={
                "in_channels": 4,
                "out_channels": 4,
                "feature_channels": 72,
                "upscale": 4,
                "bias": True,
                "train_mode": False,
                "num_blocks": 20,
            },
            srclass="supers2.models.opensr_baseline.cnn.CNNSR",
        ),
        "sr__opensrbaseline__cnn__expanded__l1": AvailableModel(
            parameters={
                "in_channels": 4,
                "out_channels": 4,
                "feature_channels": 96,
                "upscale": 4,
                "bias": True,
                "train_mode": False,
                "num_blocks": 24,
            },
            srclass="supers2.models.opensr_baseline.cnn.CNNSR",
        ),
        # SWIN Models
        "sr__opensrbaseline__swin__lightweight__l1": AvailableModel(
            parameters={
                "img_size": (128, 128),
                "in_channels": 4,
                "out_channels": 4,
                "embed_dim": 72,
                "depths": [4, 4, 4, 4],
                "num_heads": [4, 4, 4, 4],
                "window_size": 4,
                "mlp_ratio": 2.0,
                "upscale": 4,
                "resi_connection": "1conv",
                "upsampler": "pixelshuffledirect",
            },
            srclass="supers2.models.opensr_baseline.swin.Swin2SR",
        ),
        "sr__opensrbaseline__swin__small__l1": AvailableModel(
            parameters={
                "img_size": (128, 128),
                "in_channels": 4,
                "out_channels": 4,
                "embed_dim": 96,
                "depths": [6] * 6,
                "num_heads": [6] * 6,
                "window_size": 8,
                "mlp_ratio": 2.0,
                "upscale": 4,
                "resi_connection": "1conv",
                "upsampler": "pixelshuffle",
            },
            srclass="supers2.models.opensr_baseline.swin.Swin2SR",
        ),
        "sr__opensrbaseline__swin__medium__l1": AvailableModel(
            parameters={
                "img_size": (128, 128),
                "in_channels": 4,
                "out_channels": 4,
                "embed_dim": 120,
                "depths": [8] * 8,
                "num_heads": [8] * 8,
                "window_size": 8,
                "mlp_ratio": 4.0,
                "upscale": 4,
                "resi_connection": "1conv",
                "upsampler": "pixelshuffle",
            },
            srclass="supers2.models.opensr_baseline.swin.Swin2SR",
        ),
        "sr__opensrbaseline__swin__expanded__l1": AvailableModel(
            parameters={
                "img_size": (64, 64),
                "in_channels": 4,
                "out_channels": 4,
                "embed_dim": 192,
                "depths": [8] * 8,
                "num_heads": [8] * 8,
                "window_size": 4,
                "mlp_ratio": 4.0,
                "upscale": 4,
                "resi_connection": "1conv",
                "upsampler": "pixelshuffle",
            },
            srclass="supers2.models.opensr_baseline.swin.Swin2SR",
        ),
        # MAMBA Models
        "sr__opensrbaseline__mamba__lightweight__l1": AvailableModel(
            parameters={
                "img_size": (128, 128),
                "in_channels": 4,
                "out_channels": 4,
                "embed_dim": 32,
                "depths": [4, 4, 4, 4],
                "num_heads": [4, 4, 4, 4],
                "mlp_ratio": 2,
                "upscale": 4,
                "window_size": 4,
                "attention_type": "sigmoid_02",
                "upsampler": "pixelshuffledirect",
                "resi_connection": "1conv",
                "operation_attention": "sum",
            },
            srclass="supers2.models.opensr_baseline.mamba.MambaSR",
        ),
        "sr__opensrbaseline__mamba__small__l1": AvailableModel(
            parameters={
                "img_size": (128, 128),
                "in_channels": 4,
                "out_channels": 4,
                "embed_dim": 64,
                "depths": [6, 6, 6, 6],
                "num_heads": [6, 6, 6],
                "mlp_ratio": 2,
                "upscale": 4,
                "attention_type": "sigmoid_02",
                "upsampler": "pixelshuffle",
                "resi_connection": "1conv",
                "operation_attention": "sum",
            },
            srclass="supers2.models.opensr_baseline.mamba.MambaSR",
        ),
        "sr__opensrbaseline__mamba__medium__l1": AvailableModel(
            parameters={
                "img_size": (128, 128),
                "in_channels": 4,
                "out_channels": 4,
                "embed_dim": 96,
                "depths": [8, 8, 8, 8, 8, 8],
                "num_heads": [8, 8, 8, 8, 8, 8],
                "mlp_ratio": 4,
                "upscale": 4,
                "attention_type": "sigmoid_02",
                "upsampler": "pixelshuffle",
                "resi_connection": "1conv",
                "operation_attention": "sum",
            },
            srclass="supers2.models.opensr_baseline.mamba.MambaSR",
        ),
        "sr__opensrbaseline__mamba__expanded__l1": AvailableModel(
            parameters={
                "img_size": (128, 128),
                "in_channels": 4,
                "out_channels": 4,
                "embed_dim": 120,
                "depths": [8, 8, 8, 8, 8, 8],
                "num_heads": [8, 8, 8, 8, 8, 8],
                "mlp_ratio": 4,
                "upscale": 4,
                "attention_type": "sigmoid_02",
                "upsampler": "pixelshuffle",
                "resi_connection": "1conv",
                "operation_attention": "sum",
            },
            srclass="supers2.models.opensr_baseline.mamba.MambaSR",
        ),
        "sr__opensrdiffusion__large__l1": AvailableModel(
            parameters={"upscale": 4},
            srclass="supers2.models.opensr_diffusion.main.SRmodel",
        ),
        "sr__opensrdiffusion__large__l1_v2": AvailableModel(
            parameters={"upscale": 4},
            srclass="supers2.models.opensr_diffusion.main.SRmodel",
        ),
        # Zero-parameter Models
        "sr__simple__bilinear": AvailableModel(
            parameters={"upscale": 4},
            srclass="supers2.models.simple.BilinearSR",
        ),
        "sr__simple__bicubic": AvailableModel(
            parameters={"upscale": 4},
            srclass="supers2.models.simple.BicubicSR",
        ),
    }
)


fusionx2models = AvailableModels(
    object={
        # CNN Models
        "fusionx2__opensrbaseline__cnn__lightweight__l1": AvailableModel(
            parameters={
                "in_channels": 10,
                "out_channels": 6,
                "feature_channels": 24,
                "upscale": 1,
                "bias": True,
                "train_mode": False,
                "num_blocks": 6,
            },
            srclass="supers2.models.opensr_baseline.cnn.CNNSR",
        ),
        "fusionx2__opensrbaseline__cnn__small__l1": AvailableModel(
            parameters={
                "in_channels": 10,
                "out_channels": 6,
                "feature_channels": 48,
                "upscale": 1,
                "bias": True,
                "train_mode": True,
                "num_blocks": 16,
            },
            srclass="supers2.models.opensr_baseline.cnn.CNNSR",
        ),
        "fusionx2__opensrbaseline__cnn__medium__l1": AvailableModel(
            parameters={
                "in_channels": 10,
                "out_channels": 6,
                "feature_channels": 72,
                "upscale": 1,
                "bias": True,
                "train_mode": True,
                "num_blocks": 20,
            },
            srclass="supers2.models.opensr_baseline.cnn.CNNSR",
        ),
        "fusionx2__opensrbaseline__cnn__expanded__l1": AvailableModel(
            parameters={
                "in_channels": 10,
                "out_channels": 6,
                "feature_channels": 96,
                "upscale": 1,
                "bias": True,
                "train_mode": False,
                "num_blocks": 24,
            },
            srclass="supers2.models.opensr_baseline.cnn.CNNSR",
        ),
        "fusionx2__opensrbaseline__cnn__large__l1": AvailableModel(
            parameters={
                "in_channels": 10,
                "out_channels": 6,
                "feature_channels": 150,
                "upscale": 1,
                "bias": True,
                "train_mode": False,
                "num_blocks": 36,
            },
            srclass="supers2.models.opensr_baseline.cnn.CNNSR",
        ),
    }
)


fusionx4models = AvailableModels(
    object={
        # CNN Models
        "fusionx4__opensrbaseline__cnn__lightweight__l1": AvailableModel(
            parameters={
                "in_channels": 10,
                "out_channels": 6,
                "feature_channels": 24,
                "upscale": 1,
                "bias": True,
                "train_mode": False,
                "num_blocks": 6,
            },
            srclass="supers2.models.opensr_baseline.cnn.CNNSR",
        ),
        "fusionx4__opensrbaseline__cnn__small__l1": AvailableModel(
            parameters={
                "in_channels": 10,
                "out_channels": 6,
                "feature_channels": 48,
                "upscale": 1,
                "bias": True,
                "train_mode": True,
                "num_blocks": 16,
            },
            srclass="supers2.models.opensr_baseline.cnn.CNNSR",
        ),
        "fusionx4__opensrbaseline__cnn__medium__l1": AvailableModel(
            parameters={
                "in_channels": 10,
                "out_channels": 6,
                "feature_channels": 72,
                "upscale": 1,
                "bias": True,
                "train_mode": True,
                "num_blocks": 20,
            },
            srclass="supers2.models.opensr_baseline.cnn.CNNSR",
        ),
        "fusionx4__opensrbaseline__cnn__expanded__l1": AvailableModel(
            parameters={
                "in_channels": 10,
                "out_channels": 6,
                "feature_channels": 96,
                "upscale": 1,
                "bias": True,
                "train_mode": False,
                "num_blocks": 24,
            },
            srclass="supers2.models.opensr_baseline.cnn.CNNSR",
        ),
        "fusionx4__opensrbaseline__cnn__large__l1": AvailableModel(
            parameters={
                "in_channels": 10,
                "out_channels": 6,
                "feature_channels": 150,
                "upscale": 1,
                "bias": True,
                "train_mode": False,
                "num_blocks": 36,
            },
            srclass="supers2.models.opensr_baseline.cnn.CNNSR",
        ),
    }
)


AllModels = AvailableModels(
    object={
        **SRmodels.object,
        **fusionx2models.object,
        **fusionx4models.object,
    }
)
