cnn_lightweight = {
    "in_channels": 10,
    "out_channels": 6,
    "feature_channels": 24,
    "upscale": 1,
    "bias": True,
    "train_mode": True,
    "num_blocks": 6,
}

cnn_small = {
    "in_channels": 10,
    "out_channels": 6,
    "feature_channels": 48,
    "upscale": 1,
    "bias": True,
    "train_mode": True,
    "num_blocks": 16,
}

cnn_medium = {
    "in_channels": 10,
    "out_channels": 6,
    "feature_channels": 72,
    "upscale": 1,
    "bias": True,
    "train_mode": True,
    "num_blocks": 20,
}

cnn_expanded = {
    "in_channels": 10,
    "out_channels": 6,
    "feature_channels": 96,
    "upscale": 1,
    "bias": True,
    "train_mode": True,
    "num_blocks": 24,
}

swin_lightweight = {
    "img_size": (128, 128),
    "in_channels": 10,
    "out_channels": 6,
    "embed_dim": 72,
    "depths": [4, 4, 4, 4],
    "num_heads": [4, 4, 4, 4],
    "window_size": 4,
    "mlp_ratio": 2.0,
    "upscale": 1,
    "resi_connection": "1conv",
    "upsampler": "pixelshuffledirect",
}


swin_small = {
    "img_size": (128, 128),
    "in_channels": 10,
    "out_channels": 6,
    "embed_dim": 96,
    "depths": [6] * 6,
    "num_heads": [6] * 6,
    "window_size": 8,
    "mlp_ratio": 2.0,
    "upscale": 1,
    "resi_connection": "1conv",
    "upsampler": "pixelshuffle",
}


swin_medium = {
    "img_size": (128, 128),
    "in_channels": 10,
    "out_channels": 6,
    "embed_dim": 120,
    "depths": [8] * 8,
    "num_heads": [8] * 8,
    "window_size": 8,
    "mlp_ratio": 4.0,
    "upscale": 1,
    "resi_connection": "1conv",
    "upsampler": "pixelshuffle",
}


swin_expanded = {
    "img_size": (64, 64),
    "in_channels": 10,
    "out_channels": 6,
    "embed_dim": 192,
    "depths": [8] * 8,
    "num_heads": [8] * 8,
    "window_size": 4,
    "mlp_ratio": 4.0,
    "upscale": 1,
    "resi_connection": "1conv",
    "upsampler": "pixelshuffle",
}

mamba_lightweight = {
    "img_size": (128, 128),
    "in_channels": 10,
    "out_channels": 6,
    "embed_dim": 32,
    "depths": [4, 4, 4, 4],
    "num_heads": [4, 4, 4, 4],
    "mlp_ratio": 2,
    "upscale": 1,
    "window_size": 4,
    "attention_type": "sigmoid_02",
    "upsampler": "pixelshuffledirect",
    "resi_connection": "1conv",
    "operation_attention": "sum",
}


mamba_small = {
    "img_size": (128, 128),
    "in_channels": 10,
    "out_channels": 6,
    "embed_dim": 64,
    "depths": [6, 6, 6, 6],
    "num_heads": [6, 6, 6],
    "mlp_ratio": 2,
    "upscale": 1,
    "attention_type": "sigmoid_02",
    "upsampler": "pixelshuffle",
    "resi_connection": "1conv",
    "operation_attention": "sum",
}


mamba_medium = {
    "img_size": (128, 128),
    "in_channels": 10,
    "out_channels": 6,
    "embed_dim": 96,
    "depths": [8, 8, 8, 8, 8, 8],
    "num_heads": [8, 8, 8, 8, 8, 8],
    "mlp_ratio": 4,
    "upscale": 1,
    "attention_type": "sigmoid_02",
    "upsampler": "pixelshuffle",
    "resi_connection": "1conv",
    "operation_attention": "sum",
}


mamba_expanded = {
    "img_size": (128, 128),
    "in_channels": 10,
    "out_channels": 6,
    "embed_dim": 120,
    "depths": [8, 8, 8, 8, 8, 8],
    "num_heads": [8, 8, 8, 8, 8, 8],
    "mlp_ratio": 4,
    "upscale": 1,
    "attention_type": "sigmoid_02",
    "upsampler": "pixelshuffle",
    "resi_connection": "1conv",
    "operation_attention": "sum",
}
