import numpy as np
import matplotlib.pyplot as plot
from PIL import Image
import scipy.misc as sc
import os
from os.path import join, exists
import pdb


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


def fetch_batch(batch_iter, batch_size, images_list, num_classes, num_channels, H=256, W=256):
    ip_imgs = [];labels = [];names = []
    instances = np.zeros((num_classes))

    for idx in range(batch_iter*batch_size, (batch_iter+1)*batch_size):
        img_name = images_list[idx]

        gt_labels = np.array(Image.open(join('./data/labels', f'{img_name}.png')).resize((H, W), Image.NEAREST))
        for x in range(H):
            for y in range(W):
                # To convert last channel to background/positive
                # when its denoted by 255
                if gt_labels[x,y] == 255:
                    gt_labels[x,y] = num_classes-1
                instances[gt_labels[x, y]] += 1

        if num_channels == 1:
            img = np.array(Image.open(join('./data/orig', f'{img_name}.jpg')).convert('L').resize((H, W)))
            img = np.expand_dims(img, axis=-1)
        else:
            img = np.array(Image.open(join('./data/orig', f'{img_name}.jpg')).resize((H, W)))

        ip_imgs.append(img)
        labels.append(gt_labels)
        names.append(img_name)

    ip_imgs = np.array(ip_imgs)
    labels = np.array(labels)

    class_weights = []
    background_ratio = instances[0]/np.sum(instances)
    others_ratio = (1-background_ratio)/(num_classes-1)
    class_weights.append(background_ratio)
    for n in range(1, num_classes):
        class_weights.append(others_ratio)

    return ip_imgs, labels, class_weights, names
