import os
from tqdm import tqdm
import cv2
import numpy as np

class data_generator():
    # path for images
    TRAIN_HOTDOG = "hotdog-nothotdog/train/hotdog"       # 1500 images
    TRAIN_NOTHOTDOG = "hotdog-nothotdog/train/nothotdog" # 1500 images
    TEST_HOTDOG = "hotdog-nothotdog/test/hotdog"         # 322 images
    TEST_NOTHOTDOG = "hotdog-nothotdog/test/nothotdog"   # 322 images

    # data lists, each index is list containing a grayscale image and correct label
    training_data = []
    testing_data = []

    # count images
    count_train_hotdog = 0
    count_train_nothotdog = 0
    count_test_hotdog = 0
    count_test_nothotdog = 0

    @classmethod # class methods modify class state that applies across all instances of the class
    def gen_training_data(cls):
        LABELS = {cls.TRAIN_HOTDOG: 0, cls.TRAIN_NOTHOTDOG: 1} # hotdogs set to 0, else 1
        for label in LABELS:
            for file in tqdm(os.listdir(label)): # create list from files in label, and iterates through list
                file_path = os.path.join(label, file)
                image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE) # load a color image in grayscale
                cls.training_data.append([np.array(image), LABELS[label]]) # group image to correct label

                # count data to keep balance
                if label == cls.TRAIN_HOTDOG:
                    cls.count_train_hotdog += 1
                if label == cls.TRAIN_NOTHOTDOG:
                    cls.count_train_nothotdog += 1

        np.random.shuffle(cls.training_data)
        np.save("data/training_data.npy", cls.training_data) # save data to a .npy file

        # output count of each image
        print("Training hotdogs:", cls.count_train_hotdog)
        print("Training not hotdogs:", cls.count_train_nothotdog)

    @classmethod # class methods modify class state that applies across all instances of the class
    def gen_testing_data(cls):
        LABELS = {cls.TEST_HOTDOG: 0, cls.TEST_NOTHOTDOG: 1} # hotdogs set to 0, else 1
        for label in LABELS:
            for file in tqdm(os.listdir(label)): # create list from files in label, and iterates through list
                file_path = os.path.join(label, file)
                image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE) # load a color image in grayscale
                cls.testing_data.append([np.array(image), LABELS[label]]) # group image to correct label

                # count data to keep balance
                if label == cls.TEST_HOTDOG:
                    cls.count_test_hotdog += 1
                if label == cls.TEST_NOTHOTDOG:
                    cls.count_test_nothotdog += 1

        np.random.shuffle(cls.testing_data)
        np.save("data/testing_data.npy", cls.testing_data) # save data to a .npy file

        # output count of each image
        print("Testing hotdogs:", cls.count_test_hotdog)
        print("Testing not hotdogs:", cls.count_test_nothotdog)

data_generator.gen_training_data()
data_generator.gen_testing_data()
