import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from rembg import remove
from PIL import Image
import io

def read_image(image_path):
    with open(image_path, 'rb') as img_file:
        img_bytes = img_file.read()
    return img_bytes

def remove_background(img_bytes):
    img_no_bg_bytes = remove(img_bytes)
    img_no_bg = Image.open(io.BytesIO(img_no_bg_bytes))
    return img_no_bg

def convert_to_numpy(img_no_bg):
    img = np.array(img_no_bg)
    white_background = np.all(img == [0, 0, 0, 0], axis=-1)
    img[white_background] = [255, 255, 255, 255]
    return img

def convert_to_rgb(img):
    image = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    return image

def reshape_pixels(image):
    pixels = image.reshape(-1, 3)
    return pixels

def convert_to_lab(image):
    image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
    pixels_lab = image_lab.reshape(-1, 3)
    return pixels_lab

def cluster_pixels(pixels, k):
    kmeans = KMeans(n_clusters=k)
    labels = kmeans.fit_predict(pixels)
    quant = kmeans.cluster_centers_.astype("uint8")[labels]
    return kmeans.cluster_centers_

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

    plt.show()

# path_to_image = 'download.jpeg'
# dominant_colors_rgb, dominant_colors_lab, no_bg_image = find_dominant_colors_rgb_lab(path_to_image)
# plot_dominant_colors(path_to_image, no_bg_image, dominant_colors_rgb, dominant_colors_lab)