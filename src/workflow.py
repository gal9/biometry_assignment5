import os
import numpy as np
from skimage.feature import local_binary_pattern

from src.data import read_grayscale_image
from src.features import flatten_image, LBP_interpolation, LBP_uniform, LBP_histogram
from src.recognition import euclidian_distance_metric_recognition, TP_rate_from_distances


def workflow(radius_neighbors_pairs=[], methode="LBP", width: int = 128, height: int = 128):
    # Automated testing of different parameters
    data_location = "data/cropped"

    processed = [[] for _ in radius_neighbors_pairs]

    c = 0

    # Loop through data directory and transform all the images
    for filename in os.listdir(data_location):
        f = os.path.join(data_location, filename)

        if os.path.isdir(f):
            processed_tmp = [[] for _ in radius_neighbors_pairs]

            for image_file in os.listdir(f):
                image_location = os.path.join(f, image_file)

                if(image_file.endswith(".png")):
                    # print(f"Reading image {image_location}", end="\r")
                    c += 1
                    print(f"Reading image {c}; {image_location}", end="\r")

                    # Reading grayscale image and transforming it to 1D vector
                    image_grayscale = read_grayscale_image(image_location, width, height)
                    # Compute the required feature vecotr
                    if(methode == "LBP"):
                        for i, pair in enumerate(radius_neighbors_pairs):
                            processed_tmp[i].append(flatten_image(LBP_interpolation(image_grayscale, pair[0], pair[1], width,
                                                                                    height)))
                    elif(methode == "LBP_uniform"):
                        for i, pair in enumerate(radius_neighbors_pairs):
                            processed_tmp[i].append(flatten_image(LBP_uniform(image_grayscale, pair[0], pair[1], width, height)))
                    elif(methode == "LBP_histogram_8x8"):
                        for i, pair in enumerate(radius_neighbors_pairs):
                            processed_tmp[i].append(LBP_histogram(image_grayscale, pair[0], pair[1], 8, 8, width, height))
                    elif(methode == "LBP_histogram_4x4"):
                        for i, pair in enumerate(radius_neighbors_pairs):
                            processed_tmp[i].append(LBP_histogram(image_grayscale, pair[0], pair[1], 4, 4, width, height))
                    elif(methode == "LBP_histogram_16x16"):
                        for i, pair in enumerate(radius_neighbors_pairs):
                            processed_tmp[i].append(LBP_histogram(image_grayscale, pair[0], pair[1], 16, 16, width, height))
                    elif(methode == "LBP_histogram_32x32"):
                        for i, pair in enumerate(radius_neighbors_pairs):
                            processed_tmp[i].append(LBP_histogram(image_grayscale, pair[0], pair[1], 32, 32, width, height))
                    elif(methode == "sklearn_lbp"):
                        for i, pair in enumerate(radius_neighbors_pairs):
                            processed_tmp[i].append(flatten_image(local_binary_pattern(image_grayscale, R=pair[0], P=pair[1])))
                    elif(methode == "pixel_by_pixel"):
                        processed_tmp[0].append(flatten_image(image_grayscale))

            for i, pair in enumerate(radius_neighbors_pairs):
                processed[i].append(np.array(processed_tmp[i]))

    processed = np.array(processed)

    # Saving results to file
    print()
    print("Results for method " + methode)

    with open('readme.txt', 'a') as f:
        f.write("Results for method " + methode + "\n")

    # Loop through all hyperparameters
    for i, pair in enumerate(radius_neighbors_pairs):
        # np.save(f"results/{methode}_{pair[0]}_{pair[1]}", processed[i])
        # Calculate eucludean distances
        distances_euclidean = euclidian_distance_metric_recognition(processed[i], metric="euclidean")
        # Calculate cosine distances
        distances_cosin = euclidian_distance_metric_recognition(processed[i], metric="cosine")
        # calculate rank-1 for euclidean
        tp_rate_euclidean = TP_rate_from_distances(distances_euclidean)
        # calculate rank-1 for cosine
        tp_rate_cosin = TP_rate_from_distances(distances_cosin)

        # Save results
        with open('readme.txt', 'a') as f:
            f.write(f"Radius: {pair[0]}; neighbors: {pair[1]} => euclidean: {tp_rate_euclidean}; cosine: {tp_rate_cosin} \n")
            f.flush()

        print(f"Radius: {pair[0]}; neighbors: {pair[1]} =>  euclidean: {tp_rate_euclidean}; cosine: {tp_rate_cosin} ")
