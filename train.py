import tensorflow as tf
import numpy as np
import os
from time import time
from functions import sparse_weighted_cost1, sparse_unweighted_cost, pixel_wise_softmax
from data_utils import save_image, load_data, zero_mean_batch, batch_class_weights
from UNet import unet

current_dir = os.getcwd()
num_classes = 22
learning_rate = 1e-3
epochs = 100
batch_size = 1
num_channels = 1
num_batches = 91

##### Computational Graph - Start #####

X = tf.placeholder(tf.float32, shape=(batch_size, 256, 256, num_channels), name="myInput")
sparse_label = tf.placeholder(tf.int32, shape=(batch_size, 256, 256), name="myOutput")
class_weights = tf.placeholder(tf.float32, shape=(num_classes), name="class_weights")

global_step = tf.Variable(0, "global_step")

net_logits = unet(X, num_classes)

summ1 = tf.summary.scalar('unweighted_cost', sparse_unweighted_cost(net_logits, sparse_label, num_classes))
summ2 = tf.summary.scalar('weighted_cost', sparse_weighted_cost1(net_logits, sparse_label, class_weights, num_classes))

learning_rate_node = tf.train.exponential_decay(
    learning_rate=learning_rate,
    global_step=global_step, decay_steps=100,
    decay_rate=0.95, staircase=True)

optimizer = tf.train.GradientDescentOptimizer(
    learning_rate=learning_rate_node).minimize(sparse_weighted_cost1(net_logits, sparse_label, class_weights, num_classes))

summ = tf.summary.merge_all()

##### Computational Graph - End #####

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter('/UNet/log/model_graph/', sess.graph)
    saver = tf.train.Saver()

    print([X.shape, sparse_label.shape, net_logits.shape])
    print(tf.trainable_variables())

    counter = 0

    for e in range(epochs):

        start_time = time()

        for bn in range(num_batches):

            # Load batch data using data_utils
            x_train, y_train = load_data("./d/Carvana", bn)
            if len(x_train.shape) == 3:
                x_train = np.expand_dims(x_train, axis=3)

            # Zero mean the batch
            x_train = x_train/255
            x_train = zero_mean_batch(x_train)

            # Check shapes & save images when doing test run
            if bn==0:
                print(f"shape of x_train is = {x_train.shape}")
                print(f"shape of y_train is = {y_train.shape}")
                for name_count, image in enumerate(x_train):
                    # print(image.shape)
                    save_image(image=image, name=f"image_batch1_{name_count}")

            # Calculate class weights in a batch
            c_weights = batch_class_weights(y_train, num_classes)

            # print(f"true background prediction = {np.mean(y_train[0][0, :, :, 0])}")
            # print(f"true boundaries prediction = {np.mean(y_train[0][0, :, :, 21])}")
            for i in range(len(x_train)):
                logits = sess.run(net_logits, feed_dict={X: np.expand_dims(x_train[i], axis=0)})
                # print(f"shape of net output is = {logits.shape}")

                image, _ = sess.run([pixel_wise_softmax(net_logits), optimizer],
                                    feed_dict={X: np.expand_dims(x_train[i], axis=0),
                                               sparse_label: np.expand_dims(y_train[i], axis=0),
                                               class_weights: c_weights})

                if counter%10 == 0:
                    summary = sess.run(summ, feed_dict={X: np.expand_dims(x_train[i], axis=0),
                                                        sparse_label: np.expand_dims(y_train[i], axis=0),
                                                        class_weights: c_weights})
                    writer.add_summary(summary, counter)

                # if i == 0:
                #     print(f"avg background prediction = {np.mean(image[0, :, :, 0])}")
                #     print(f"avg boundaries prediction = {np.mean(image[0, :, :, 21])}")

                if bn==0 and e%2==0:
                    # print(image.shape)
                    save_image(f"image{i}.png", image)

                counter += 1

        end_time = time()
        save_path = saver.save(sess, "./tmp/without_batch/model.ckpt")
        duration = end_time - start_time
        print(f"epoch no. {e} : done in {duration} sec")

    writer.close()
