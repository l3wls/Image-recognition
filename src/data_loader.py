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

def _parse_labels(path):
    """Read a label file and return a list of integer labels."""
    with open(path, 'r') as f:
        return [int(line.strip()) for line in f if line.strip() != '']


def load_digit_training():
    images_path = os.path.join(DATA_DIR, 'digitdata', 'trainingimages')
    labels_path = os.path.join(DATA_DIR, 'digitdata', 'traininglabels')
    images = _parse_images(images_path, DIGIT_ROWS, DIGIT_COLS)
    labels = _parse_labels(labels_path)
    assert len(images) == len(labels), (
        f"Digit train: {len(images)} images but {len(labels)} labels"
    )
    return images, labels


def load_digit_validation():
    images_path = os.path.join(DATA_DIR, 'digitdata', 'validationimages')
    labels_path = os.path.join(DATA_DIR, 'digitdata', 'validationlabels')
    images = _parse_images(images_path, DIGIT_ROWS, DIGIT_COLS)
    labels = _parse_labels(labels_path)
    assert len(images) == len(labels), (
        f"Digit validation: {len(images)} images but {len(labels)} labels"
    )
    return images, labels

def load_face_training():
    images_path = os.path.join(DATA_DIR, 'facedata', 'facedatatrain')
    labels_path = os.path.join(DATA_DIR, 'facedata', 'facedatatrainlabels')
    images = _parse_images(images_path, FACE_ROWS, FACE_COLS)
    labels = _parse_labels(labels_path)
    assert len(images) == len(labels), (
        f"Face train: {len(images)} images but {len(labels)} labels"
    )
    return images, labels


def load_face_validation():
    images_path = os.path.join(DATA_DIR, 'facedata', 'facedatavalidation')
    labels_path = os.path.join(DATA_DIR, 'facedata', 'facedatavalidationlabels')
    images = _parse_images(images_path, FACE_ROWS, FACE_COLS)
    labels = _parse_labels(labels_path)
    assert len(images) == len(labels), (
        f"Face validation: {len(images)} images but {len(labels)} labels"
    )
    return images, labels