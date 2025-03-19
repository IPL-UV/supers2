# 

<p align="center">
  <img src="assets/images/logo_ss2.png" width="30%">
</p>

<p align="center">
   <em>A Python package for enhancing the spatial resolution of Sentinel-2 satellite images up to 2.5 meters</em> 🚀
</p>


<p align="center">
<a href='https://pypi.python.org/pypi/supers2'>
    <img src='https://img.shields.io/pypi/v/supers2.svg' alt='PyPI' />
</a>
<a href="https://opensource.org/licenses/MIT" target="_blank">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License">
</a>
<a href="https://github.com/psf/black" target="_blank">
    <img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Black">
</a>
<a href="https://pycqa.github.io/isort/" target="_blank">
    <img src="https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336" alt="isort">
</a>
<a href="https://colab.research.google.com/drive/1TD014aY145q1reKN644egUtIM6tIx9vH?usp=sharing" target="_blank">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
</p>


---

**GitHub**: [https://github.com/IPL-UV/supers2](https://github.com/IPL-UV/supers2) 🌐
**PyPI**: [https://pypi.org/project/supers2/](https://pypi.org/project/supers2/) 🛠️

---

## **Table of Contents**

- [**Overview** 🌍](#overview-)
- [**Installation** ⚙️](#installation-)
- [**How to use** 🛠️](#how-to-use-)
  - [**Load libraries**](#load-libraries)
  - [**Download Sentinel-2 L2A cube**](#download-sentinel-2-l2a-cube)
  - [**Prepare the data (CPU and GPU usage)**](#prepare-the-data-cpu-and-gpu-usage)
  - [**Default model setup**](#default-model-setup)
  - [**Configuring Model**](#configuring-model)
- [**Available Models:**](#available-models)
  - [**1. CNN Models**](#1-cnn-models)
  - [**2. SWIN Models**](#2-swin-models)
  - [**3. MAMBA Models**](#3-mamba-models)
  - [**4. Diffusion Model**](#4-diffusion-model)
  - [**5. Simple Models (Bilinear and Bicubic)**](#5-simple-models-bilinear-and-bicubic)
- [**Predict only RGBNIR bands**](#predict-only-rgbnir-bands)
- [**Estimate the uncertainty of the model** 📊](#estimate-the-uncertainty-of-the-model-)
- [**Estimate the Local Attention Map of the model** 📊](#estimate-the-local-attention-map-of-the-model-)



## **Overview** 🌍

**supers2** is a Python package designed to enhance the spatial resolution of Sentinel-2 satellite images to 2.5 meters using a set of neural network models. 

## **Installation** ⚙️

Install the latest version from PyPI:

```bash
pip install supers2
```

From GitHub:

```bash
pip install git+https://github.com/IPL-UV/supers2.git
```

## **How to use** 🛠️

### **Load libraries**

```python
import matplotlib.pyplot as plt
import numpy as np
import torch
import cubo

import supers2

```

### **Download Sentinel-2 L2A cube**

```python
# Create a Sentinel-2 L2A data cube for a specific location and date range
da = cubo.create(
    lat=39.49152740347753,
    lon=-0.4308725142800361,
    collection="sentinel-2-l2a",
    bands=["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"],
    start_date="2023-01-01",
    end_date="2023-12-31",
    edge_size=64,
    resolution=10
)
```

### **Prepare the data (CPU and GPU usage)**

When converting a NumPy array to a PyTorch tensor:

- **GPU:** Use `.cuda()` to transfer the tensor to the GPU if available, improving speed for large datasets or models.

- **CPU:** If no GPU is available, PyTorch defaults to the CPU; omit `.cuda()`.

Here’s how you can handle both scenarios dynamically:

```python
# Check if CUDA is available, use GPU if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Converting data to a PyTorch tensor ensures efficient computation and compatibility, while scaling standardizes pixel values to improve performance.

```python
# Convert the data array to NumPy and scale
original_s2_numpy = (da[11].compute().to_numpy() / 10_000).astype("float32")

# Create the tensor and move it to the appropriate device (CPU or GPU)
X = torch.from_numpy(original_s2_numpy).float().to(device)
```

### **Download and Load the model**

```python
import mlstac

# Download the model
mlstac.download(
  file="https://huggingface.co/tacofoundation/supers2/resolve/main/simple_model/mlm.json",
  output_dir="models2/CNN_Light_SR",
)

# Load the model
model = mlstac.load("models/supers2_simple_model").compiled_model()
model = model.to(device)

# Apply model
superX = model(X[None]).squeeze(0)
```

The first plot shows the original Sentinel-2 RGB image (10m resolution). The second plot displays the enhanced version with finer spatial details (2.5m resolution) using a lightweight CNN.

```python
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(X[[2, 1, 0]].permute(1, 2, 0).cpu().numpy()*4)
ax[0].set_title("Original S2")
ax[1].imshow(superX[[2, 1, 0]].permute(1, 2, 0).cpu().numpy()*4)
ax[1].set_title("Enhanced Resolution S2")
plt.show()
```

<p align="center">
  <img src="assets/images/first_plot.png" width="100%">
</p>


## **Predict only RGBNIR bands**

```python
superX = supers2.predict_rgbnir(X[[2, 1, 0, 6]])
```

### Estimate the Local Attention Map of the model 📊


```python
kde_map, complexity_metric, robustness_metric, robustness_vector = supers2.lam(
    X=X[[2, 1, 0, 6]].cpu(), # The input tensor
    model=models.srx4, # The SR model
    h=240, # The height of the window
    w=240, # The width of the window
    window=128, # The window size
    scales = ["1x", "2x", "3x", "4x", "5x", "6x", "7x", "8x"]
)

# Visualize the results
plt.imshow(kde_map)
plt.title("Kernel Density Estimation")
plt.show()

plt.plot(robustness_vector)
plt.title("Robustness Vector")
plt.show()
```

<p align="center">
  <img src="assets/images/kernel.png" width="50%">
</p>
<br>
<p align="center">
  <img src="assets/images/vector.png" width="70%">
</p>

<!-- ## Use the opensr-test and supers2 to analyze the hallucination pixels 📊 -->