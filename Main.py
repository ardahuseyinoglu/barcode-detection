import glob
from LineDetection import detectLinesOnBarcode

# get all image address which are original images and ground truth images
original_image_addresses = glob.glob('dataset\Original_Subset\*.*')
ground_truth_image_addresses = glob.glob('dataset\Detection_Subset\*.*')
number_of_image = len(original_image_addresses)

# detect and draw lines for each image
for image_index in range(number_of_image):
    original_image_address = original_image_addresses[image_index]
    ground_truth_image_address = ground_truth_image_addresses[image_index]
    detectLinesOnBarcode(original_image_address, ground_truth_image_address)
