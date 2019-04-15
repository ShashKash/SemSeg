import numpy as np
import matplotlib.pyplot as plot
from PIL import Image


def save_image(name, image):
    # name must be a string
    indexes = len(image.shape)
    if indexes == 3 and image.shape[2] == 1:
        height = image.shape[0]
        width = image.shape[1]
        img = np.zeros((height, width))
        for i in range(0, height):
            for j in range(0, width):
                img[i, j] = image[i, j][0] * 12
        plot.imsave(name, img)
    elif indexes == 4 and image.shape[3] != 1:
        height = image.shape[1]
        width = image.shape[2]
        # print(height)
        # print(width)
        img = np.zeros((height, width))
        for i in range(0, height):
            for j in range(0, width):
                img[i, j] = (np.argmax(image[0][i, j], axis=0) * 12)
        plot.imsave(name, img, cmap='prism')


def load_data(path, batch_num):
    # path must be a string
    labels = np.load(f"{path}/train_y_sparse/y_labels{batch_num}.npy")
    x_input = np.load(f"{path}/train_x/x_orig{batch_num}.npy")
    return x_input, labels


def zero_mean_batch(batch):
    meaned_batch = batch
    mean = np.mean(batch, axis=0)
    # print(f"mean of the batch is = {mean}")
    for i in range(len(batch)):
        meaned_batch[i] = batch[i] - mean
    return meaned_batch


def batch_class_weights(batch, classes):
    instances = np.zeros((classes))
    class_weights = []
    for image in batch:
        for i in range(batch.shape[1]):
            for j in range(batch.shape[2]):
                instances[image[i, j]] += 1

    background_ratio = instances[0]/np.sum(instances)
    others_ratio = (1-background_ratio)/(classes-1)
    class_weights.append(background_ratio)
    for n in range(1, classes):
        class_weights.append(others_ratio)
    print(class_weights)
    return class_weights


def load_image(path, b_n_w=False):
    if b_n_w:
        img = np.array(Image.open(path).resize(256, 256))
    else:
        img = np.array(Image.open(path).convert('L').resize((256, 256)))
        b_n_w = True

    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=2)
        img = np.expand_dims(img, axis=0)
    elif len(img.shape) == 3:
        img = np.expand_dims(img, axis=0)
    return img
