import cv2
import numpy as np
import matplotlib.pyplot as plt
from HoughTransform import HoughTransform
from Util import getCosSinThetaValues


def detectLinesOnBarcode(original_image_address, ground_truth_image_address):
    # read original and ground truth images
    original_image = cv2.imread(original_image_address)
    ground_truth_image = cv2.imread(ground_truth_image_address)
    # convert original image to gray scaled image
    original_image_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    # find edge points on gray scaled image by using canny edge detector
    original_image_edge_map = cv2.Canny(original_image_gray, 200, 400)
    # find the edge points appeared only in the barcode area
    edge_point_coordinates_in_barcode = determineEdgePointsInBarcode(ground_truth_image, original_image_edge_map)
    # lines keep the information of the max-voted coordinates (rho, theta) in hough space
    lines = HoughTransform(edge_point_coordinates_in_barcode, original_image_edge_map)
    # draw detected lines to original and ground truth image
    original_image_copy = original_image.copy()
    original_image_line_detected = drawLinesToImage(original_image, lines)
    ground_truth_image_line_detected = drawLinesToImage(ground_truth_image, lines)
    # concatenate images horizontally and show output
    showOutput(original_image_copy, original_image_edge_map, original_image_line_detected, ground_truth_image_line_detected)


# check each detected edge point on original image if it takes place in barcode area by the using groun-truth image
# since just the lines in the barcode area will be detected, we just need edge points in that area.
def determineEdgePointsInBarcode(ground_truth_image, image_edge_map):
    ground_truth_image_gray = cv2.cvtColor(ground_truth_image, cv2.COLOR_BGR2GRAY)
    barcode_edge_points_row_array = []
    barcode_edge_points_column_array = []
    # find the locations of edge points whose intensities are non-zero, 255
    edge_points_row_array, edge_points_column_array = np.nonzero(image_edge_map)
    for edge_point in range(len(edge_points_row_array)):
        # check if edge points is in the barcode area(white pixels in the ground-truth image)
        if ground_truth_image_gray[edge_points_row_array[edge_point]][edge_points_column_array[edge_point]] == 255:
            barcode_edge_points_row_array.append(edge_points_row_array[edge_point])
            barcode_edge_points_column_array.append(edge_points_column_array[edge_point])
    # convert them to numpy array
    barcode_edge_points_row_array = np.array(barcode_edge_points_row_array)
    barcode_edge_points_column_array = np.array(barcode_edge_points_column_array)
    return barcode_edge_points_row_array, barcode_edge_points_column_array


# draw detected lines to original image and ground-truth image
def drawLinesToImage(image, lines):
    row_coordinates_max_voted = lines[0]
    column_coordinates_max_voted = lines[1]
    thetas, cos_thetas, sin_thetas = getCosSinThetaValues()
    index_arrangment = round(lines[2].shape[0]/2)
    for i in range(len(row_coordinates_max_voted)):
        theta = column_coordinates_max_voted[i]
        rho = row_coordinates_max_voted[i] - index_arrangment
        pt1 = (int(rho*cos_thetas[theta] + 1000*(-sin_thetas[theta])), int(rho*sin_thetas[theta] + 1000*(cos_thetas[theta])))
        pt2 = (int(rho*cos_thetas[theta] - 1000*(-sin_thetas[theta])), int(rho*sin_thetas[theta] - 1000*(cos_thetas[theta])))
        cv2.line(image, pt1, pt2, (0, 0, 255), 1, cv2.LINE_AA)
    return image


# concatenate 4 image horizontally and show them
def showOutput(original_image, original_image_edge_map, original_image_line_detected, ground_truth_image_line_detected):
    # increase dimension and length of the gray scale image which is img_canny_edge to be able to combine with other 3d images
    original_image_edge_map_3d = cv2.cvtColor(original_image_edge_map, cv2.COLOR_GRAY2BGR)
    # concatenate images horizontally
    plot_image = np.concatenate((original_image, original_image_edge_map_3d, original_image_line_detected, ground_truth_image_line_detected), axis=1)
    # show the output image
    plot_image = cv2.cvtColor(plot_image, cv2.COLOR_BGR2RGB)
    plt.imshow(plot_image)
    plt.show()



