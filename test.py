import cv2 as cv
from src.support_functions import bounding_box_from_mask, crop_ear

persone = "183"
image = "01.png"

image_path = "./data/images/" + persone + "/" + image
mask_path = "./data/masks/" + persone + "/" + image

gt_box = bounding_box_from_mask(mask_path)

image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
# image = cv.rectangle(image, (gt_box[0], gt_box[1]), (gt_box[2], gt_box[3]), (255, 0, 0), 2)
image = crop_ear(image, gt_box)
# image = image[0:1, 0:1]
cv.imwrite("test.png", image)
