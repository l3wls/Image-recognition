import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

DIGIT_ROWS = 28
DIGIT_COLS = 28

FACE_ROWS = 70
FACE_COLS = 60


def _parse_images(path, rows, cols):
    """Read a packed text image file and return a list of 2D char grids."""
    with open(path, 'r') as f:
        lines = f.readlines()

    images = []
    total_lines = len(lines)
    i = 0
    while i + rows <= total_lines:
        grid = []
        for r in range(rows):
            row = lines[i + r].rstrip('\n')
            row = row.ljust(cols)[:cols]
            grid.append(list(row))
        images.append(grid)
        i += rows
    return images