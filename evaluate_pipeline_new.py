import torch
import cv2 as cv
import os
import time
from src.recognition import recognize
from src.data import read_grayscale_image
from src.train import train
from src.support_functions import crop_ear

def best_prediction(prediction_list):
    """
    Predicts the 
    """
    best = ("", 1000)
    
    for prediction in prediction_list:
        score = sum([e[1] for e in prediction_list if e[0]==prediction[0]])/len([e for e in prediction_list if e[0]==prediction[0]])

        if(score < best[1]):
            best = prediction
    
    return best[0]

start = time.time()
train()
print(f"Finished enrollment in {time.time()-start}")

# Detection model
model = torch.hub.load('./src/yolov5/yolov5', 'custom', path='./yolo5s.pt', source="local")


test_folder = "./data/images_test"
mojority_in = [5, 10, 15, 20]
mojority_in_count = [0, 0, 0, 0, ]

count_all = 0


# Loop through data directory and transform all the images
for folder_number in os.listdir(test_folder):
    persone_folder_path = os.path.join(test_folder, folder_number)

    if os.path.isdir(persone_folder_path):
        for image_file in os.listdir(persone_folder_path):
            count_all += 1

            image_location = os.path.join(persone_folder_path, image_file)
            image_grayscale = read_grayscale_image(image_location, 128, 128)
            
            # Detect ear
            results = model(image_location).xyxy[0].numpy()
            # Crop image
            image_grayscale = crop_ear(image_grayscale, results[0])

            predictions = recognize(image_grayscale, "./data/trained_gt")

            people_ranked = [path[0][-10:-7] for path in predictions]

            for i, m in enumerate(majority_in):
                # prediction = best_prediction[people_ranked[:m]]
                try:
                    prediction = mode(people_ranked[:m])
                except:
                    prediction = people_ranked[0]

                print(f"Sample {image_location} recognized as {prediction}")

                if(prediction == folder_number):
                    majority_in_count[i] += 1

print(f"majority in 5: {majority_in_count[0]/count_all}")
print(f"majority in 10: {majority_in_count[1]/count_all}")
print(f"majority in 15: {majority_in_count[2]/count_all}")
print(f"majority in 20: {majority_in_count[3]/count_all}")
