from PIL import Image
import os
from scipy import misc
import numpy as np
import pickle

# collect all the images and reduce their resoultion to a constant size

desired_size = 64
size = 64, 64
example_data = []


def reduce_resolution(directory, save_dir):
    for img in os.listdir(directory):
        im = Image.open(directory + img)

        old_size = im.size  # old_size[0] is in (width, height) format
        ratio = float(desired_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        im.thumbnail(size)

        new_im = Image.new("RGB", (desired_size, desired_size))
        new_im.paste(im,  ((desired_size-new_size[0])//2,
                    (desired_size-new_size[1])//2))

        new_im.save(save_dir + img, "JPEG")


def average_pixel(pixel):
    return np.average(pixel)


def read_image(directory, label):
    for img in os.listdir(directory):
        ls = {"features": [], "label": []}
        image = misc.imread(directory + img)
        w, h = Image.open(directory + img).size

        for x in range(0, w):
            for y in range(0, h):
                ls["features"].append(image[x][y][0])
                ls["features"].append(image[x][y][1])
                ls["features"].append(image[x][y][2])
                #print(ls["features"])

        ls["label"] = label
        #print(len(ls["features"]))
        example_data.append(ls)

    #for example in example_data:
        #print(example)


def get_training_data():
    pickle_file = open("training_obj.pickle", "rb")
    data = pickle.load(pickle_file)
    return data


def get_testing_data():
    pickle_file = open("test_obj.pickle", "rb")
    data = pickle.load(pickle_file)
    return data



#train preprocessing
#reduce_resolution("./dataset/train/hot_dog/", "./dataset/train_red/hot_dog/")
#reduce_resolution("./dataset/train/not_hot_dog/", "./dataset/train_red/not_hot_dog/")

#test preprocessing
#reduce_resolution("./dataset/test/hot_dog/", "./dataset/test_red/hot_dog/")
#reduce_resolution("./dataset/test/not_hot_dog/", "./dataset/test_red/not_hot_dog/")

#train prepare data set
#read_image("./dataset/train_red/hot_dog/", [1, 0])
#read_image("./dataset/train_red/not_hot_dog/", [0, 1])

#test prepare data set
#read_image("./dataset/test_red/hot_dog/", 0)
#read_image("./dataset/test_red/not_hot_dog/", 1)

#save training data set
#shuffle everything
#np.random.shuffle(example_data)
#pickle_file = open("training_obj.pickle", "wb")
#pickle.dump(example_data, pickle_file)
#pickle_file.close()

#save test data set
#np.random.shuffle(example_data)
#pickle_file = open("test_obj.pickle", "wb")
#pickle.dump(example_data, pickle_file)
#pickle_file.close()

