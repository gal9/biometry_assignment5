import cv2 as cv
from src.support_functions import bounding_box_from_mask, crop_ear
from src.data import read_grayscale_image
from src.recognition import recognize

"""persone = "183"
image = "01.png"

image_path = "./data/images/" + persone + "/" + image
mask_path = "./data/masks/" + persone + "/" + image

gt_box = bounding_box_from_mask(mask_path)

image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
# image = cv.rectangle(image, (gt_box[0], gt_box[1]), (gt_box[2], gt_box[3]), (255, 0, 0), 2)
image = crop_ear(image, gt_box)
# image = image[0:1, 0:1]
cv.imwrite("test.png", image)"""

image = read_grayscale_image("./data/cropped_test_gt/181/14.png", 128, 128)

pred = recognize(image, "./data/trained_gt")

print(pred)

