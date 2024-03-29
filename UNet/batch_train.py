import tensorflow as tf
import numpy as np
import os
from os.path import join, exists
from time import time
from functions import sparse_weighted_cost, pixel_wise_softmax, tf_get_mask
from data_utils import save_image, fetch_batch
from UNet import unet
import json
import pprint

current_dir = os.getcwd()
dataset = 'carvana'
learning_rate = 1e-4
epochs = 5
batch_size = 1
num_channels = 1
H = 256
W = 256
print_n = 10  # every print_n iters save summaries

if dataset == 'pascal':
    num_classes = 22
    with open(join(current_dir, 'data/train.json')) as fp:
        images_list = json.load(fp)
    fp.close()
elif dataset == 'carvana':
    num_classes = 2
    images_list = np.load('./Carvana/cars.npy')

print(len(images_list))
num_batches = len(images_list)//8 - 1

##### Computational Graph - Start #####

X = tf.placeholder(tf.float32, shape=(batch_size, H, W, num_channels), name="myInput")
sparse_label = tf.placeholder(tf.int32, shape=(batch_size, H, W), name="myOutput")
class_weights = tf.placeholder(tf.float32, shape=(num_classes), name="class_weights")

global_step = tf.Variable(0, "global_step")

net_logits = unet(X, num_classes)

# This prediction for each pixel are probabilities of it being in a class across the channels
pred_probs = pixel_wise_softmax(net_logits)

pred_mask = tf_get_mask(batch_size, H, W, pred_probs, num_classes)

# unweighted_loss = sparse_unweighted_cost(net_logits, sparse_label, num_classes)
unweighted_loss, weighted_loss = sparse_weighted_cost(net_logits, sparse_label, class_weights, num_classes)
summ_unweighted = tf.summary.scalar('unweighted_cost', unweighted_loss)
summ_weighted = tf.summary.scalar('weighted_cost', weighted_loss)
summ_pred_mask = tf.summary.image('pred_mask', pred_mask)

learning_rate_node = tf.train.exponential_decay(
    learning_rate=learning_rate,
    global_step=global_step, decay_steps=150,
    decay_rate=0.95, staircase=True)

optimizer = tf.train.GradientDescentOptimizer(
    learning_rate=learning_rate_node).minimize(weighted_loss)

summ = tf.summary.merge_all()

##### Computational Graph - End #####

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter(join(current_dir, 'logs'), sess.graph)
    saver = tf.train.Saver()

    print([X.shape, sparse_label.shape, net_logits.shape])
    pprint.pprint(tf.trainable_variables())

    counter = 0

    for e in range(epochs):

        start_time = time()

        for bn in range(num_batches):

            # Load batch data using data_utils
            x_train, y_train, c_weights, names = fetch_batch(dataset, bn, batch_size, images_list, num_classes, num_channels)
            # Shape of x_train = (batch_size, H, W, num_channels)
            # print(x_train.shape)
            # Shape of y_train = (batch_size, H, W)
            # print(y_train.shape)

            # 0 Normalize the batch
            if batch_size != 1:
                x_train = x_train - np.mean(x_train, axis=0)
                x_train = x_train/np.std(x_train, axis=0)

            # Save some images when doing test run
            # if bn==0:
            #     for name_count, image in enumerate(x_train):
            #         # print(image.shape)
            #         save_image(image=image, name=f"image_batch1_{name_count}")

            logits, _ = sess.run([net_logits, optimizer],
                                 feed_dict={X: x_train,
                                            sparse_label: y_train,
                                            class_weights: c_weights})

            if bn % print_n == 0:
                loss, summary = sess.run([weighted_loss, summ],
                                         feed_dict={X: x_train,
                                                    sparse_label: y_train,
                                                    class_weights: c_weights})
                print(loss)
                writer.add_summary(summary, counter)

            # if bn==0 and e%2==0:
            #     # print(image.shape)
            #     save_image(f"image{i}.png", image)

            counter += 1

        end_time = time()
        save_path = saver.save(sess, join(current_dir, "checkpoints/model.ckpt"))
        duration = end_time - start_time
        print(f"epoch no. {e} : done in {duration} sec")

    writer.close()
