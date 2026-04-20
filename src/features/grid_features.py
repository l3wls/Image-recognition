import numpy as np

def extract_grid_features(image, grid_rows=4, grid_cols=4):
    img_rows = len(image)
    img_cols = len(image[0]) if img_rows > 0 else 0
    features = np.zeros(grid_rows * grid_cols, dtype=np.float64)

    for r in range(grid_rows):
        for c in range(grid_cols):
            row_start = r * img_rows // grid_rows
            row_end = (r + 1) * img_rows // grid_rows
            col_start = c * img_cols // grid_cols
            col_end = (c + 1) * img_cols // grid_cols

            found = any(
                image[row][col] != ' '
                for row in range(row_start, row_end)
                for col in range(col_start, col_end)
            )
            features[r * grid_cols + c] = 1 if found else 0

    return features
