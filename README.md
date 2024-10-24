# 

<p align="center">
  <img src="./assets/images/banner_supers2.png" width="50%">
</p>

<p align="center">
   <em>A Python package for enhancing the spatial resolution of Sentinel-2 satellite images to 2.5 meters</em> 🚀
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

## **Overview** 📊

**supers2** is a Python package designed to enhance the spatial resolution of Sentinel-2 satellite images to 2.5 meters using advanced neural network models. It facilitates downloading (cubo package), preparing, and processing the Sentinel-2 data and applies deep learning models to enhance the spatial resolution of the imagery.

## **Installation** ⚙️

Install the latest version from PyPI:

```bash
pip install cubo supers2
```

## **How to use** 🛠️

### **Basic usage: enhancing spatial resolution of Sentinel-2 images** 🌍

#### **Load libraries**

```python
import cubo
import supers2
import torch
import numpy as np
```

#### **Download Sentinel-2 L2A cube**

```python
# Create a Sentinel-2 L2A data cube for a specific location and date range
da = cubo.create(
    lat=4.31, 
    lon=-76.2, 
    collection="sentinel-2-l2a", 
    bands=["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"], 
    start_date="2021-06-01", 
    end_date="2021-10-10", 
    edge_size=128, 
    resolution=10
)
```

#### **Prepare the data (CPU and GPU usage)**

When converting the NumPy array to a PyTorch tensor, the use of `cuda()` is optional and depends on whether the user has access to a GPU. Below is the explanation for both cases:

- **GPU:** If a GPU is available and CUDA is installed, you can transfer the tensor to the GPU using `.cuda()`. This improves the processing speed, especially for large datasets or deep learning models.

- **CPU:** If no GPU is available, the tensor will be processed on the CPU, which is the default behavior in PyTorch. In this case, simply omit the `.cuda()` call.

Here’s how you can handle both scenarios dynamically:

```python
# Convert the data array to NumPy and scale
original_s2_numpy = (da[11].compute().to_numpy() / 10_000).astype("float32")

# Check if CUDA is available, use GPU if possible, otherwise fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the tensor and move it to the appropriate device (CPU or GPU)
X = torch.from_numpy(original_s2_numpy).float().to(device)
```

#### **Define the resolution enhancement model**
```python
# Set up the model to enhance the spatial resolution
models = supers2.setmodel(
    SR_model_loss="l1", 
    SR_model_name="cnn", 
    SR_model_size="small", 
    Fusionx2_model_size="lightweight", 
    Fusionx4_model_size="lightweight"
)
```
### **Apply spatial resolution enhancement**

```python
# Apply the model to enhance the image resolution to 2.5 meters
superX = supers2.predict(X, models=models, resolution="2.5m")
```

### **Visualize the results** 🎨

#### **Display images**

```python
import matplotlib.pyplot as plt

# Plot the original and enhanced-resolution images
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(X[[2, 1, 0]].permute(1, 2, 0).cpu().numpy()*4)
ax[0].set_title("Original S2")
ax[1].imshow(superX[[2, 1, 0]].permute(1, 2, 0).cpu().numpy()*4)
ax[1].set_title("Enhanced Resolution S2")
plt.show()
```

<p align="center">
  <img src="./assets/images/example1.png" width="100%">
</p>

## **Supported features and filters** ✨

- **Enhance spatial resolution to 2.5 meters:** Use advanced CNN models to enhance Sentinel-2 imagery.
- **Neural network-based approach:** Integration of multiple model sizes to fit different computing needs (small, lightweight).
- **Python integration:** Easily interact with data cubes through the Python API, supporting seamless workflows.