import numpy as np
import math


def flatten_image(image):
    # Transforms a 2D image array to 1D vecotr"
    return np.array(image).flatten()


def get_coordinates(radius: int, neighbors: int, round_coords: bool):
    # Calculates coordinates for the specified radius and neighbour number
    coordinates = []

    for p in range(neighbors):
        # Set cooridnates for every neighbour based on the equations
        coordinates.append((-math.sin((p*2*math.pi)/neighbors)*radius,
                           math.cos((p*2*math.pi)/neighbors)*radius))

    # Round the coordinates to integers if specified (LBP without interpolation)
    if(round_coords):
        coordinates = [(round(x), round(y)) for x, y in coordinates]

    return coordinates


def LBP_value(image, center_value, coordinates):
    # Calculate LBP value based on cooridnates and center pixel value
    lbp = 0
    for i, point in enumerate(coordinates):
        # First coordinate represents row (y axis) and second represents column (x axis)
        pixel_value = image[point[0]][point[1]]

        # If value of pixel is larger than the center value add 2**i
        if(center_value <= pixel_value):
            lbp += 2**i

    return lbp


def LBP(image, radius: int, neighbors: int, width: int, height: int):
    # Calculate LBP values without interpolation (round the coorinates of neighbours)
    coordinates = get_coordinates(radius, neighbors, True)

    values = []
    # Loop through all rows
    for row_i, row in enumerate(image):
        row_values = []
        # Loop thourgh every row
        for pixel_i, pixel in enumerate(row):
            # If pixel in question is on the edge set value to 0
            if(row_i < radius or pixel_i < radius or row_i+radius >= width or pixel_i+radius >= height):
                row_values.append(0)
            else:
                # Calculate actual coordinates of neghbours of center pixel
                neighbor_coordinates = [(row_i+y, pixel_i+x) for x, y in coordinates]
                # Claculate LBP value and save it
                row_values.append(LBP_value(image, pixel, neighbor_coordinates))

        values.append(row_values)

    return np.array(values)


def LBP_value_interpolation(image, center_value, coordinates):
    # Calculate LBP value based on cooridnates and center pixel value
    lbp = 0
    # Loop thourgh all neighbours
    for i, point in enumerate(coordinates):
        # First coordinate represents row (y axis) and second represents column (x axis)

        # calculate the values of the neighbouring pixels (if coordinates are integers it also works
        # as we devide by four at the end)
        pixle_value = 0
        pixle_value += image[math.ceil(point[0])][math.ceil(point[1])]
        pixle_value += image[math.ceil(point[0])][math.floor(point[1])]
        pixle_value += image[math.floor(point[0])][math.ceil(point[1])]
        pixle_value += image[math.floor(point[0])][math.floor(point[1])]
        pixle_value /= 4

        # If value of pixel is larger than the center value add 2**i
        if(center_value <= pixle_value):
            lbp += 2**i

    return lbp


def LBP_interpolation(image, radius: int, neighbors: int, width: int, height: int):
    # Calculate LBP values without interpolation
    coordinates = get_coordinates(radius, neighbors, False)

    values = []
    # Loop through all rows
    for row_i, row in enumerate(image):
        row_values = []
        # Loop thourgh every row
        for pixel_i, pixel in enumerate(row):
            # If pixel in question is on the edge set value to 0
            if(row_i < radius or pixel_i < radius or row_i+radius >= height or pixel_i+radius >= width):
                row_values.append(0)
            else:
                # Calculate actual coordinates of neghbours of center pixel
                neighbor_coordinates = [(row_i+y, pixel_i+x) for x, y in coordinates]
                # Claculate LBP value (with interpolation) and save it
                row_values.append(LBP_value_interpolation(image, pixel, neighbor_coordinates))

        values.append(row_values)

    return np.array(values)


def LBP_histogram(image, radius: int, neighbors: int, columns: int, rows: int, width: int, height: int):
    # First calculate LBP values
    LBP_picture = LBP_interpolation(image, radius, neighbors, width, height)

    # calculate grid parameters
    height = len(image)
    width = len(image[0])

    grid_height = int(height/rows)
    grid_width = int(width/columns)

    final_hist = np.array([])

    # Loop through grid and calculate histogram for every piece
    for column in range(columns):
        for row in range(rows):
            # Extract piece (subarray)
            subarray = LBP_picture[row*grid_height:(row+1)*grid_height,
                                   column*grid_width:(column+1)*grid_width]

            # Calculate histogram
            hist = np.histogram(subarray, bins=range(257))

            # Append it to final feature vector
            final_hist = np.concatenate((final_hist, hist[0]), axis=0)

    return final_hist


# Lookup table for uniform LBP (LBP value transformation)
lookup_table = [0, 1, 2, 3, 4, 58, 5, 6, 7, 58, 58, 58, 8, 58, 9, 10, 11, 58, 58, 58, 58, 58, 58, 58, 12, 58, 58,
                58, 13, 58, 14, 15, 16, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 17, 58, 58, 58, 58, 58,
                58, 58, 18, 58, 58, 58, 19, 58, 20, 21, 22, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
                58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 23, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
                58, 58, 58, 58, 24, 58, 58, 58, 58, 58, 58, 58, 25, 58, 58, 58, 26, 58, 27, 28, 29, 30, 58, 31, 58, 58, 58,
                32, 58, 58, 58, 58, 58, 58, 58, 33, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 34, 58, 58,
                58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
                58, 58, 35, 36, 37, 58, 38, 58, 58, 58, 39, 58, 58, 58, 58, 58, 58, 58, 40, 58, 58, 58, 58, 58, 58, 58, 58, 58,
                58, 58, 58, 58, 58, 58, 41, 42, 43, 58, 44, 58, 58, 58, 45, 58, 58, 58, 58, 58, 58, 58, 46, 47, 48, 58, 49,
                58, 58, 58, 50, 51, 52, 58, 53, 54, 55, 56, 57]


def LBP_uniform(image, radius: int, neighbors: int, width: int, height: int):
    # First calculate LBP values
    LBP_picture = LBP_interpolation(image, radius, neighbors, width, height)

    # Loop through LBP values
    for row_i, row in enumerate(LBP_picture):
        for pixel_i, pixel in enumerate(row):
            # Change LBP value based on the lookup table value
            LBP_picture[row_i][pixel_i] = lookup_table[pixel]

    return LBP_picture
