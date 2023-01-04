import cv2 as cv
import numpy as np

def bounding_box_from_mask(maks_path):
    """
    Returns a box of shape [left_bottom_x, left_bottom_y, right_top_x, right_top_y]
    """
    image = np.array(cv.imread(maks_path, cv.IMREAD_GRAYSCALE))

    height = len(image)
    width = len(image[0])

    high_row = None
    low_row = None
    right_col = 0
    left_col = width

    for i, row in enumerate(image):
        for j, el in enumerate(row):
            if(el != 0):
                # Assign rows
                low_row = i
                if(high_row is None):
                    high_row = i

                # Test if columns are left or right maximized
                if(j<left_col):
                    left_col = j
                if(j>right_col):
                    right_col = j

    return [left_col, low_row, right_col, high_row]


def crop_ear(image, box):

    return image[box[3]:box[1]+1, box[0]:box[2]+1]
