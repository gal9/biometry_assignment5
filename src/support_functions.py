import cv2 as cv
import numpy as np
import os

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

    return image[round(box[3]):round(box[1])+1, round(box[0]):round(box[2])+1]

def crop_images(data_location= "./data/images_train", cropped_location= "./data/cropped_train", boxes_location= "./data/boxes_trained"):

    for filename in os.listdir(data_location):
        f = os.path.join(data_location, filename)

        # Check if dir exists and create it otherwise
        if(not os.path.isdir(os.path.join(cropped_location, filename))):
            os.makedirs(os.path.join(cropped_location, filename))

        if os.path.isdir(f):
            for image_file in os.listdir(f):

                if(image_file.endswith(".png")):
                    file_name = image_file[:-4] + ".txt"
                    box_location = os.path.join(boxes_location, filename, file_name)
                    image_location = os.path.join(data_location, filename, image_file)
                    cropped_location_single = os.path.join(cropped_location, filename, image_file)
                    
                    if(os.path.isfile(box_location)):
                        # Read the box
                        with open(box_location, "r") as f_box:
                            lines = f_box.readlines()

                            box = [int(round(float(el))) for el in lines[0].split()]

                        image = cv.imread(image_location, cv.IMREAD_GRAYSCALE)
                        
                        image = crop_ear(image, box)
                        print(cropped_location_single)
                        cv.imwrite(cropped_location_single, image)
