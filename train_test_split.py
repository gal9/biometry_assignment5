import os
import shutil

folder = "./data/cropped_gt"
train_folder = "./data/cropped_train_gt"
test_folder = "./data/cropped_test_gt"


# Check if dir exists and create it otherwise
if(not os.path.isdir(train_folder)):
    os.makedirs(train_folder)
if(not os.path.isdir(test_folder)):
    os.makedirs(test_folder)

# Loop through data directory and transform all the images
for folder_number in os.listdir(folder):
    f = os.path.join(folder, folder_number)

    if os.path.isdir(f):
        to_train = int(len(os.listdir(f))/2)
        c = 0

        for image_file in os.listdir(f):
            image_location = os.path.join(f, image_file)

            if(c<to_train):
                dst = os.path.join(train_folder, folder_number)
            else:
                dst = os.path.join(test_folder, folder_number)

            # create folder of a persone if it does not exists            
            if(not os.path.isdir(dst)):
                os.makedirs(dst)

            dst = os.path.join(dst, image_file)
            
            shutil.copyfile(image_location, dst)
            c += 1