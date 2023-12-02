# Image Clustering

This project is a Python application that uses machine learning to find the dominant colors in an image. It uses the KMeans clustering algorithm to cluster the colors in the image and then displays the dominant colors in both RGB and LAB color spaces.

## Requirements

The project requires the following Python packages:
- streamlit
- opencv-python-headless
- numpy
- scikit-learn
- rembg
- Pillow
- matplotlib

## Usage

The main application is a Streamlit app. You can run the app using the following command:
```
streamlit run src/app.py
```

In the app, you can upload an image and choose the number of clusters. The app will then display the dominant colors in the image.

```python
import streamlit as st
# Other imports...

st.title('Image Clustering')

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image_path = uploaded_file.name

    st.image(image_path, caption='Uploaded Image.', use_column_width=True)

    k = st.slider('Number of clusters', min_value=2, max_value=20, value=10)

    if st.button('Cluster'):
        dominant_colors_rgb, dominant_colors_lab, img_no_bg = find_dominant_colors_rgb_lab(image_path, k)

        plot_dominant_colors(image_path, img_no_bg, dominant_colors_rgb, dominant_colors_lab)
```

## Modules

The project includes a module for image processing and clustering:

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from rembg import remove
from PIL import Image
import io

# Functions...

def find_dominant_colors_rgb_lab(image_path, k=10):
    # Function implementation...
    return dominant_colors_rgb, dominant_colors_lab, img

# Other functions...

# Tests...
```

This module includes functions to read an image, remove the background, convert the image to a numpy array, convert the image to RGB and LAB color spaces, reshape the pixels for clustering, and cluster the pixels using KMeans.

## Tests

The project includes unit tests for the functions in the cluster module:

```python
import unittest

class TestCluster(unittest.TestCase):
    # Test cases...
    pass
```

You can run the tests using the following command:

```
python -m unittest tests/test_cluster.py
```

## Example

```python
path_to_image = 'download.jpeg'
dominant_colors_rgb, dominant_colors_lab, no_bg_image = find_dominant_colors_rgb_lab(path_to_image)
plot_dominant_colors(path_to_image, no_bg_image, dominant_colors_rgb, dominant_colors_lab)
```

Here is an example of how to use the cluster module to find the dominant colors in an image.

This will display a plot with the original image, the image with the background removed, and the dominant colors in RGB and LAB color spaces.

## Docker

To run the application using Docker, follow these steps:

1. Build the Docker image:
```
docker build -t image_cluster .
```
2. Run the Docker container:
```
docker run -p 8501:8501 image_cluster
```
The application will be accessible at `http://localhost:8501`.
