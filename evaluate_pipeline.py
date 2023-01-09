import torch
import cv2 as cv
import os
import time
from src.recognition import recognize
from src.data import read_grayscale_image
from src.train import train
from src.support_functions import crop_ear

start = time.time()
train()
print(f"Finished enrollment in {time.time()-start}")

test_folder = "./data/images_test"
# Detection model
model = torch.hub.load('./src/yolov5/yolov5', 'custom', path='./yolo5s.pt', source="local")

count_all = 0
tp_rank1 = 0
tp_rank5 = 0
tp_rank10 = 0
tp_rank15 = 0
tp_rank20 = 0

# Loop through people
for folder_number in os.listdir(test_folder):
    persone_folder_path = os.path.join(test_folder, folder_number)

    if os.path.isdir(persone_folder_path):
        # Loop through images
        for image_file in os.listdir(persone_folder_path):
            count_all += 1

            # Read image
            image_location = os.path.join(persone_folder_path, image_file)
            image_grayscale = read_grayscale_image(image_location, 128, 128)
            # Detect ear
            results = model(image_location).xyxy[0].numpy()
            print(result[0])
            # Crop image
            image_grayscale = crop_ear(image_grayscale, results[0])

            print(image_grayscale)
            cv.imshow("n", image_grayscale)

            prediction = recognize(image_grayscale, "./data/trained_gt")

            people_ranked = [path[0][-10:-7] for path in prediction]

            print(f"Sample {image_location} recognized as {people_ranked[0]}")

            if(people_ranked[0] == folder_number):
                tp_rank1 += 1
                tp_rank5 += 1
                tp_rank10 += 1
                tp_rank15 += 1
                tp_rank20 += 1
            elif(folder_number in people_ranked[:5]):
                tp_rank5 += 1
                tp_rank10 += 1
                tp_rank15 += 1
                tp_rank20 += 1
            elif(folder_number in people_ranked[:10]):
                tp_rank10 += 1
                tp_rank15 += 1
                tp_rank20 += 1
            elif(folder_number in people_ranked[:15]):
                tp_rank15 += 1
                tp_rank20 += 1
            elif(folder_number in people_ranked[:20]):
                tp_rank20 += 1            

print(f"rank-1: {tp_rank1/count_all}")
print(f"rank-5: {tp_rank5/count_all}")
print(f"rank-10: {tp_rank10/count_all}")
print(f"rank-15: {tp_rank15/count_all}")
print(f"rank-20: {tp_rank20/count_all}")