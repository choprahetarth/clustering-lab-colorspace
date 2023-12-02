import streamlit as st
import cv2
import numpy as np
from sklearn.cluster import KMeans
from rembg import remove
from PIL import Image
import io
import matplotlib.pyplot as plt
from cluster import read_image, remove_background, convert_to_numpy, convert_to_rgb, reshape_pixels, convert_to_lab, cluster_pixels

def find_dominant_colors_rgb_lab(image_path, k=10):
    img_bytes = read_image(image_path)
    img_no_bg = remove_background(img_bytes)
    img = convert_to_numpy(img_no_bg)
    image = convert_to_rgb(img)
    pixels = reshape_pixels(image)
    pixels_lab = convert_to_lab(image)
    dominant_colors_rgb = cluster_pixels(pixels, k)
    dominant_colors_lab = cluster_pixels(pixels_lab, k)
    return dominant_colors_rgb, dominant_colors_lab, img

def plot_dominant_colors(original_image_path, no_bg_image, dominant_colors_rgb, dominant_colors_lab):
    # Read the original image
    original_image = cv2.imread(original_image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, 4, figsize=(24, 6))

    # Plot the original image
    ax[0].imshow(original_image)
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    # Plot the image with background removed
    ax[1].imshow(no_bg_image)
    ax[1].set_title('Image with Background Removed')
    ax[1].axis('off')

    # Plot the dominant colors in RGB
    ax[2].bar(range(len(dominant_colors_rgb)), [1]*len(dominant_colors_rgb), color=dominant_colors_rgb/255)
    ax[2].set_title('Dominant Colors in RGB')
    ax[2].axis('off')

    # Convert LAB to RGB
    dominant_colors_lab_rgb = [cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_Lab2RGB)[0][0] for color in dominant_colors_lab]

    # Plot the dominant colors in LAB
    ax[3].bar(range(len(dominant_colors_lab_rgb)), [1]*len(dominant_colors_lab_rgb), color=np.array(dominant_colors_lab_rgb)/255)
    ax[3].set_title('Dominant Colors in LAB')
    ax[3].axis('off')

    st.pyplot(fig)

st.title('Image Clustering')

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image_path = uploaded_file.name

    st.image(image_path, caption='Uploaded Image.', use_column_width=True)

    k = st.slider('Number of clusters', min_value=2, max_value=20, value=10)

    if st.button('Cluster'):
        dominant_colors_rgb, dominant_colors_lab, img_no_bg = find_dominant_colors_rgb_lab(image_path, k)

        plot_dominant_colors(image_path, img_no_bg, dominant_colors_rgb, dominant_colors_lab)