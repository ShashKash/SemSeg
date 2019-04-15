import tensorflow as tf
import numpy as np
import os
from time import time
from functions import pixel_wise_softmax
from data_utils import save_image, load_data, load_image
from UNet import unet

current_dir = os.getcwd()
num_classes = 2
mean = np.load("./d/Carvana/mean_of_traindata.npy")

##### Computational Graph - Start #####

X = tf.placeholder(tf.float32, shape=(1, 256, 256, 1), name="myInput")
net_logits = unet(X, num_classes)

##### Computational Graph - End #####

with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess, "./tmp/without_batch/model.ckpt")
    print("Model Loaded")

    print(tf.trainable_variables())

    # Load image as numpy array using data_utils
    img = load_image("./Carvana/train/0d53224da2b7_04.jpg")
    # Zero mean image based on training data
    img = img - np.expand_dims(np.expand_dims(mean, axis=3), axis=0)
    img = img/255
    print(img.shape)

    # TODO: Add batch prediction.

    image = sess.run(pixel_wise_softmax(net_logits), feed_dict={X: img})
    # print(f"shape of net output is = {logits.shape}")
    save_image(image=image, name="test_prediction.png")
