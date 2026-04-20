import numpy as np

def extract_counting_features(image):
    """
    Takes a 2D list of characters and returns:
    [count of spaces, count of non-space characters]
    """

    space_count = 0
    non_space_count = 0

    for row in image:
        for char in row:
            if char == ' ':
                space_count += 1
            else:
                non_space_count += 1

    return np.array([space_count, non_space_count])