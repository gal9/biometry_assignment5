import sys
from sklearn.metrics import pairwise_distances
import numpy as np


def euclidian_distance_recognition(target_image_flattened, images_flattened,
                                   target_persone_i: int, target_sample_i: int, metric: str):
    # Find out which persone has sample closest to target
    closest_sample = None
    closest_persone = None
    closest_distance = sys.float_info.max

    # Loop through the samples
    for persone_i, samples in enumerate(images_flattened):
        for sample_i, sample in enumerate(samples):
            # Compared sample has to be different from the one tested
            if(persone_i != target_persone_i or sample_i != target_sample_i):
                # Compute the distance for the selected metric
                distance = pairwise_distances([sample-target_image_flattened],
                                              metric)

                # Check if distance is lower than closest distance so far
                if(distance < closest_distance):
                    closest_distance = distance
                    closest_persone = persone_i
                    closest_sample = sample_i

    return closest_persone, closest_sample


def euclidian_distance_recognition_2(target_image_flattened, images_flattened,
                                     target_persone_i: int,
                                     target_sample_i: int):
    closest_sample = None
    closest_persone = None
    closest_distance = sys.float_info.max

    for persone_i, samples in enumerate(images_flattened):
        for sample_i, sample in enumerate(samples):
            if(persone_i != target_persone_i or sample_i != target_sample_i):
                distance = np.linalg.norm(sample-target_image_flattened)
                if(distance < closest_distance):
                    closest_distance = distance
                    closest_persone = persone_i
                    closest_sample = sample_i

    return closest_persone, closest_sample


def euclidian_distance_metric_recognition(images_flattened, metric: str):
    # Transform to np array
    images_flattened = np.array(images_flattened)
    # Reshape array
    images_flattened = images_flattened.reshape(-1, images_flattened.shape[-1])
    # Calculate and return pairwise distances
    return pairwise_distances(images_flattened, metric=metric)


def euclidian_distance_metric_recognition_histogram(images_flattened):
    return pairwise_distances(images_flattened, metric="euclidean")


def TP_rate_from_distances(distance_mtx):
    # Claculate how many samples have smallest distances to a sample of the same persone
    TP = 0
    # Loop through all the samples
    for i, row in enumerate(distance_mtx):
        closest_distance = closest_distance = sys.float_info.max
        closest_j = None
        # Find the closest sample
        for j, distance in enumerate(row):
            if(i != j and closest_distance > distance):
                closest_j = j
                closest_distance = distance
        
        # If people are the same cout one true positive
        if(int(i/10) == int(closest_j/10)):
            TP += 1

    return (TP/len(distance_mtx))
