import numpy as np

def extract_pixel_features(image):
    """
    Convert 2D image into 1D binary numpy array
    ' ' -> 0
    anything else -> 1
    """

    # Create a NumPy array using a flattened traversal of the image
    # This list comprehension loops through every row, then every pixel in that row
    # and converts each pixel into a binary value
    return np.array([

        # If the pixel is a space, treat it as background (0)
        # Otherwise, treat it as foreground (1)
        0 if pixel == ' ' else 1

        # Loop through each row in the image (2D structure)
        for row in image

        # Loop through each pixel in the current row
        for pixel in row

        # This effectively "flattens" the 2D image into a 1D sequence
    ])
