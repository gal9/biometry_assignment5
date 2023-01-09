import cv2 as cv


def area_of_intersection(minx1, miny1, maxx1, maxy1, minx2, miny2, maxx2, maxy2):
    if (min(maxx1, maxx2) < max(minx1, minx2) or min(maxy1, maxy2) < max(miny1, miny2)):
        return 0

    w = min(maxx1, maxx2) - max(minx1, minx2)
    h = min(maxy1, maxy2) - max(miny1, miny2)

    return w*h


def area_of_union(width1, height1, width2, height2, ai):
    return (width1*height1)+(width2*height2)-ai


def _iou(box1_x, box1_y, box1_width, box1_height, box2_x, box2_y, box2_width, box2_height):
    ai = area_of_intersection(box1_x, box1_y, box1_x+box1_width, box1_y+box1_height, box2_x, box2_y, box2_x+box2_width,
                              box2_y+box2_height)

    au = area_of_union(box1_width, box1_height, box2_width, box2_height, ai)

    return ai/au


def iou(boxes, true_x, true_y, true_width, true_height):
    r = []

    for box in boxes:
        r.append(_iou(true_x, true_y, true_width, true_height, box[0], box[1], box[2], box[3]))

    return r


def iou_for_yolo(boxes, true_x, true_y, true_width, true_height):
    """
    Params:
        true_x: gt x of top left corner
        true_y: gt y of top left corner
        true_width: width of box
        true_height: height of box
    """
    r = []

    for box in boxes:
        r.append(_iou(true_x, true_y, true_width, true_height, box[0], box[1], box[2]-box[0], box[3]-box[1]))

    return r


def yolo_iou(persone_id, image_id, model):
    image = cv.imread(f"data/images/{persone_id}/{image_id}.png")
    img_height = image.shape[0]
    img_width = image.shape[1]

    with open(f"data/boxes/{persone_id}/{image_id}.txt") as f:
        # Read the ground truth file (format: left_x, low_y, right_x, top_y)
        line = f.read().split()
        box_width = int(line[2]) - int(line[0])
        box_height = int(line[1]) - int(line[3])
        box_x = int(line[0])
        box_y = int(line[3])

    results = model(f"data/images/{persone_id}/{image_id}.png").xyxy[0].numpy()

    r_s = iou_for_yolo(results, box_x, box_y, box_width, box_height)

    return r_s
