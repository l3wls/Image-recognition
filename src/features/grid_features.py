import numpy as np

def extract_grid_features(image, grid_rows=8, grid_cols=8):
    # Get number of rows (height) in the image
    img_rows = len(image)

    # Get number of columns (width); handle empty image safely
    img_cols = len(image[0]) if img_rows > 0 else 0

    # Create a 1D feature vector initialized to 0s
    # Size = total number of grid cells (rows * cols)
    # float64 because we will store density (decimal values)
    features = np.zeros(grid_rows * grid_cols, dtype=np.float64)

    # Loop through each grid cell (r = row index of grid, c = column index of grid)
    for r in range(grid_rows):
        for c in range(grid_cols):
    
            # Compute the starting row index of this grid cell
            # Uses integer division to evenly split image height
            row_start = r * img_rows // grid_rows
            
            # Compute the ending row index of this grid cell
            row_end = (r + 1) * img_rows // grid_rows

            # Compute the starting column index of this grid cell
            col_start = c * img_cols // grid_cols
          
            # Compute the ending column index of this grid cell
            col_end = (c + 1) * img_cols // grid_cols

            # Count of "filled" (non-empty) pixels in this cell
            count = 0

            # Total number of pixels in this grid cell
            # Used to compute density 
            total = (row_end - row_start) * (col_end - col_start)


            # Loop through each pixel inside the current grid cell
            for row in range(row_start, row_end):
                for col in range(col_start, col_end):


                    # If pixel is NOT a space, treat it as "filled"
                    # (Assumes ' ' = background, anything else = foreground)
                    if image[row][col] != ' ':
                        count += 1  # Increment filled pixel count


            # Store the density of filled pixels in this cell
            # Convert 2D grid position (r, c) into 1D index
            # density = filled pixels / total pixels
            # If total is 0 (edge case), store 0 to avoid division error
            features[r * grid_cols + c] = count / total if total > 0 else 0

    # Return the feature vector representing the image
    return features
