import numpy as np
from Util import getCosSinThetaValues

# find locations of most voted of them in hough space
def HoughTransform(edge_point_coordinates_in_barcode, image_edge_map):
    hough_accumulator = createAccumulatorArray(image_edge_map)
    hough_accumulator = findNumberOfVotes(edge_point_coordinates_in_barcode, hough_accumulator)
    row_coordinates_max_voted, column_coordinates_max_voted = findMaxVotedLocations(hough_accumulator)
    return row_coordinates_max_voted, column_coordinates_max_voted, hough_accumulator


# structure a hough accumulator array:
def createAccumulatorArray(image_edge_map):
    # create a numpy array including theta values in radian from -pi/2 to +pi/2. it is for the x-axis(or column) of hough space
    thetas = np.deg2rad(np.arange(-90.0, 91.0))
    # create a numpy array including rho values(distance from the origin). it is for the y-axis(or row) of hough space
    # find the max value of rho which is diagonal of the image
    max_rho = int(np.ceil(np.sqrt(image_edge_map.shape[0]**2 + image_edge_map.shape[0]**2)))
    rhos = np.arange(-max_rho, max_rho + 1)
    # create accumulator array
    hough_accumulator = np.zeros((len(rhos), len(thetas)), dtype=int)
    return hough_accumulator


# voting: accumulator array keeps track for its every location how many time sinusodial curves in hough space
# belonging to each edge point in barcode area of image space pass those locations
def findNumberOfVotes(edge_coordinates_in_barcode, hough_accumulator):
    thetas, cos_thetas, sin_thetas = getCosSinThetaValues()
    index_arrangment = round(hough_accumulator.shape[0]/2)
    for theta_index in range(len(thetas)):
        # rho = x*cos(theta) + y*sin(theta)
        rho = edge_coordinates_in_barcode[1] * cos_thetas[theta_index] + edge_coordinates_in_barcode[0] * sin_thetas[theta_index]
        rho = np.round(rho)
        rho = rho.astype('int')
        for rho_index in range(len(rho)):
            hough_accumulator[rho[rho_index] + index_arrangment, theta_index] += 1
    return hough_accumulator


# find max voted locations in accumulator array.
def findMaxVotedLocations(hough_accumulator):
    hough_accumulator_flattened = hough_accumulator.flatten()
    hough_accumulator_n_max_indexes = hough_accumulator_flattened.argsort()[-30:]
    n_max_votes = hough_accumulator_flattened[hough_accumulator_n_max_indexes]
    threshold_value = n_max_votes.mean()

    # for images whose lines in barcode are detected well
    if threshold_value >= 130:
        hough_accumulator_n_max_indexes = hough_accumulator_flattened.argsort()[-40:]
        n_max_votes = hough_accumulator_flattened[hough_accumulator_n_max_indexes]
        threshold_value = n_max_votes.min()
    elif threshold_value >= 120 and threshold_value < 130:
        hough_accumulator_n_max_indexes = hough_accumulator_flattened.argsort()[-30:]
        n_max_votes = hough_accumulator_flattened[hough_accumulator_n_max_indexes]
        threshold_value = n_max_votes.min()
    elif threshold_value >= 110 and threshold_value < 120:
        hough_accumulator_n_max_indexes = hough_accumulator_flattened.argsort()[-20:]
        n_max_votes = hough_accumulator_flattened[hough_accumulator_n_max_indexes]
        threshold_value = n_max_votes.min()

    row_coordinates_max_voted, column_coordinates_max_voted = np.where(hough_accumulator >= threshold_value)
    return row_coordinates_max_voted, column_coordinates_max_voted
