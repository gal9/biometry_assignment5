import os

from src.data import read_grayscale_image
from src.features import flatten_image, LBP_interpolation, LBP_uniform, LBP_histogram


def train_recognition(methode="LBP", model_folder = "./data/trained_gt",
                      train_folder = "./data/cropped_train_gt",
                      width: int = 128, height: int = 128):    
    if(not os.path.isdir(model_folder)):
        os.makedirs(model_folder)

    # Loop through data directory and transform all the images
    for folder_number in os.listdir(train_folder):
        f = os.path.join(train_folder, folder_number)
        
        template_folder_path = os.path.join(model_folder, folder_number)

        # Create perosne folder if it does not exists
        if(not os.path.isdir(template_folder_path)):
            os.makedirs(template_folder_path)

        if os.path.isdir(f):
            for image_file in os.listdir(f):
                image_location = os.path.join(f, image_file)

                if(image_file.endswith(".png")):
                    
                    # Reading grayscale image and transforming it to 1D vector
                    image_grayscale = read_grayscale_image(image_location, width, height)
                    # Where to save the result
                    template_path = os.path.join(template_folder_path, image_file[:-4] + ".txt")                    

                    """if(methode == "LBP"):
                            processed_tmp[i].append(flatten_image(LBP_interpolation(image_grayscale, pair[0], pair[1], width,
                                                                                    height)))
                    elif(methode == "LBP_uniform"):
                        processed_tmp[i].append(flatten_image(LBP_uniform(image_grayscale, pair[0], pair[1], width, height)))
                    elif(methode == "LBP_histogram_8x8"):
                        processed_tmp[i].append(LBP_histogram(image_grayscale, pair[0], pair[1], 8, 8, width, height))
                    elif(methode == "LBP_histogram_4x4"):
                        processed_tmp[i].append(LBP_histogram(image_grayscale, pair[0], pair[1], 4, 4, width, height))"""
                    if(methode == "LBP_histogram_16x16"):
                        template = LBP_histogram(image_grayscale, 2, 8, 16, 16, width, height)
                    """elif(methode == "LBP_histogram_32x32"):
                        processed_tmp[i].append(LBP_histogram(image_grayscale, pair[0], pair[1], 32, 32, width, height))
                    elif(methode == "sklearn_lbp"):
                        processed_tmp[i].append(flatten_image(local_binary_pattern(image_grayscale, R=pair[0], P=pair[1])))
                    elif(methode == "pixel_by_pixel"):
                        processed_tmp[0].append(flatten_image(image_grayscale))"""
                    with open(template_path, "w") as template_f:
                        template_f.write(" ".join([str(el) for el in template]))