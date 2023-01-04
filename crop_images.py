import os
from src.support_functions import crop_ear
import cv2 as cv

data_location = "./data/images"
cropped_location = "./data/cropped"
boxes_location = "./data/boxes"

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

                # Read the box
                with open(box_location, "r") as f_box:
                    lines = f_box.readlines()

                    box = [int(el) for el in lines[0].split()]

                image = cv.imread(image_location, cv.IMREAD_GRAYSCALE)
                
                image = crop_ear(image, box)
                print(cropped_location_single)
                cv.imwrite(cropped_location_single, image)
                