import numpy as np

def extract_pixel_features(image):
    """
    Convert 2D image into 1D binary numpy array
    ' ' -> 0
    anything else -> 1
    """

    return np.array([
        0 if pixel == ' ' else 1
        for row in image
        for pixel in row
    ])