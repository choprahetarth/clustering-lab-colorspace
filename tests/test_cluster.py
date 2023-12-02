import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from PIL import Image
import unittest
import numpy as np
from src import cluster

class TestCluster(unittest.TestCase):

    def test_read_image(self):
        # Test that the function returns bytes
        result = cluster.read_image('download.jpeg')
        self.assertIsInstance(result, bytes)

    def test_remove_background(self):
        # Test that the function returns an Image object
        img_bytes = cluster.read_image('download.jpeg')
        result = cluster.remove_background(img_bytes)
        self.assertIsInstance(result, Image.Image)

    def test_convert_to_numpy(self):
        # Test that the function returns a numpy array
        img_bytes = cluster.read_image('download.jpeg')
        img_no_bg = cluster.remove_background(img_bytes)
        result = cluster.convert_to_numpy(img_no_bg)
        self.assertIsInstance(result, np.ndarray)

    def test_convert_to_rgb(self):
        # Test that the function returns a numpy array
        img_bytes = cluster.read_image('download.jpeg')
        img_no_bg = cluster.remove_background(img_bytes)
        img = cluster.convert_to_numpy(img_no_bg)
        result = cluster.convert_to_rgb(img)
        self.assertIsInstance(result, np.ndarray)

    def test_reshape_pixels(self):
        # Test that the function returns a 2D numpy array
        img_bytes = cluster.read_image('download.jpeg')
        img_no_bg = cluster.remove_background(img_bytes)
        img = cluster.convert_to_numpy(img_no_bg)
        image = cluster.convert_to_rgb(img)
        result = cluster.reshape_pixels(image)
        self.assertEqual(len(result.shape), 2)

    def test_convert_to_lab(self):
        # Test that the function returns a 2D numpy array
        img_bytes = cluster.read_image('download.jpeg')
        img_no_bg = cluster.remove_background(img_bytes)
        img = cluster.convert_to_numpy(img_no_bg)
        image = cluster.convert_to_rgb(img)
        result = cluster.convert_to_lab(image)
        self.assertEqual(len(result.shape), 2)

    def test_cluster_pixels(self):
        # Test that the function returns a 2D numpy array
        pixels = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result = cluster.cluster_pixels(pixels, 2)
        self.assertEqual(len(result.shape), 2)

if __name__ == '__main__':
    unittest.main()