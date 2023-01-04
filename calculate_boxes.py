import os
from src.support_functions import bounding_box_from_mask

data_location = "./data/masks"
boxes_location = "./data/boxes"

for filename in os.listdir(data_location):
    f = os.path.join(data_location, filename)

    # Check if dir exists and create it otherwise
    if(not os.path.isdir(os.path.join(boxes_location, filename))):
        os.makedirs(os.path.join(boxes_location, filename))

    if os.path.isdir(f):
        for image_file in os.listdir(f):
            image_location = os.path.join(f, image_file)

            if(image_file.endswith(".png")):

                # Calculate boxes and cast to string
                box = [str(el) for el in bounding_box_from_mask(image_location)]

                # Write results to file
                file_name = image_file[:-4] + ".txt"
                box_location = os.path.join(boxes_location, filename, file_name)
                with open(box_location, "w") as f_box:
                    f_box.write(" ".join(box))
                