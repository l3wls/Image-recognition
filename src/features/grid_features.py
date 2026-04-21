import numpy as np

def extract_grid_features(image, grid_rows=8, grid_cols=8):
    img_rows = len(image)
    img_cols = len(image[0]) if img_rows > 0 else 0
    features = np.zeros(grid_rows * grid_cols, dtype=np.float64)

    for r in range(grid_rows):
        for c in range(grid_cols):
            row_start = r * img_rows // grid_rows
            row_end = (r + 1) * img_rows // grid_rows
            col_start = c * img_cols // grid_cols
            col_end = (c + 1) * img_cols // grid_cols

            count = 0
            total = (row_end - row_start) * (col_end - col_start)

            for row in range(row_start, row_end):
                for col in range(col_start, col_end):
                    if image[row][col] != ' ':
                        count += 1

            # Density instead of binary
            features[r * grid_cols + c] = count / total if total > 0 else 0

    return features