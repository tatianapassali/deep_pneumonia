import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt
import pydicom
from pydicom.data import get_testdata_files
import pickle


def load_and_preprocess(n_train=30000, n_test=2000, image_size=128):

    # Load annotations
    annotations = np.genfromtxt('data/stage_2_train_labels.csv', delimiter=',', dtype=None, skip_header=True)
    np.random.seed(1)
    image_names = []
    labels = []
    x_min = []
    y_min = []
    width = []
    height = []

    #image_names = [x[0] for x in my_data]
    for x in annotations:
        image_names.append(x[0])
        x_min.append(x[1])
        y_min.append(x[2])
        width.append(x[3])
        height.append(x[4])
        labels.append(x[5])

    image_names = np.asarray(image_names)
    labels = np.asarray(labels)
    x_min = np.asarray(x_min)
    x_min = x_min.astype(np.int)
    y_min = np.asarray(y_min)
    y_min = y_min.astype(np.int)
    width = np.asarray(width)
    width = width.astype(np.int)
    height = np.asarray(height)
    height = height.astype(np.int)

    # Shuffle
    idx = np.random.permutation(len(labels))
    image_names = image_names[idx]
    labels = labels[idx]
    x_min = x_min[idx]
    y_min = y_min[idx]
    width = width[idx]
    height = height[idx]

    # Split the data
    test_image_names = image_names[:n_test]
    test_labels = labels[:n_test]
    test_x_min = x_min[:n_test]
    test_y_min = y_min[:n_test]
    test_width = width[:n_test]
    test_height = height[:n_test]

    train_image_names = image_names[n_test:n_train + n_test]
    train_labels = labels[n_test:n_train + n_test]
    train_x_min = x_min[n_test:n_train + n_test]
    train_y_min = y_min[n_test:n_train + n_test]
    train_width = width[n_test:n_train + n_test]
    train_height = height[n_test:n_train + n_test]


    # Actually load the data
    train_data = []
    train_masks = []

    for i in range(len(train_image_names)):
        filename = 'data/stage_2_train_images/' + str(train_image_names[i].decode("utf-8") + '.dcm')
        img = pydicom.dcmread(filename)
        img = np.asarray(img.pixel_array)
        train_mask = np.zeros((1000, 1000))
        train_mask = np.asarray(train_mask)
        if(train_labels[i]==1):
            cur_x = train_x_min[i]
            cur_y = train_y_min[i]
            w = train_width[i]
            h = train_height[i]
            train_mask[cur_y:cur_y+h, cur_x:cur_x+w] = 1

        img = resize(img, (image_size, image_size))
        train_mask = resize(train_mask, (13, 13))
        train_mask[train_mask > 0] = 1
        train_data.append(img)
        train_masks.append(train_mask)

    train_data = np.asarray(train_data)
    train_masks = np.asarray(train_masks)
    print(train_masks.shape)

    test_data = []
    test_masks = []
    for i in range(len(test_image_names)):
        filename = 'data/stage_2_train_images/' + str(test_image_names[i].decode("utf-8") + '.dcm')
        img = pydicom.dcmread(filename)
        img = np.asarray(img.pixel_array)

        test_mask = np.zeros((1000, 1000))
        test_mask = np.asarray(test_mask)

        if (train_labels[i] == 1):
            cur_x = test_x_min[i]
            cur_y = test_y_min[i]
            w = test_width[i]
            h = test_height[i]
            test_mask[cur_y:cur_y + h, cur_x:cur_x + w] = 1

        img = resize(img, (image_size, image_size))
        test_data.append(img)
        test_mask = resize(test_mask, (13, 13))
        test_mask[train_mask > 0] = 1

        test_masks.append(test_mask)

    test_data = np.asarray(test_data)
    test_masks = np.asarray(test_masks)

    # Save the preprocessed and resized images into a pickle for faster loading
    with open("dataset.pickle", "wb") as f:
        pickle.dump(train_data, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(train_labels, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(train_masks, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(test_data, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(test_labels, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(test_masks, f, pickle.HIGHEST_PROTOCOL)


def load_data():
    with open("dataset.pickle", "rb") as f:
        train_data = pickle.load(f)
        train_labels = pickle.load(f)
        train_masks = pickle.load(f)
        test_data = pickle.load(f)
        test_labels = pickle.load(f)
        test_masks = pickle.load(f)
    return train_data, train_masks, train_labels, test_data, test_labels, test_masks

def show_images(data, labels):
    for i in range(10):
        plt.figure(1)
        plt.imshow(data[i], cmap=plt.cm.bone)
        plt.show()
        print("Labels is ", labels[i])


# load_and_preprocess()
load_data()
