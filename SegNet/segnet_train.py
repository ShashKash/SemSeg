import tensorflow as tf
import numpy as np
import os
from time import time
from data_utils import save_image, load_data, zero_mean_batch, batch_class_weights
from segnet import segnet
from functions import sparse_weighted_cost, sparse_unweighted_cost, pixel_wise_softmax


current_dir = os.getcwd()
num_classes = 2
learning_rate = 0.2
epochs = 20
batch_size = 1
num_batches = 636
num_channels = 3


X = tf.placeholder(tf.float32, shape=(batch_size, 256, 256, num_channels), name="myInput")
sparse_label = tf.placeholder(tf.int32, shape=(batch_size, 256, 256), name="myOutput")
class_weights = tf.placeholder(tf.float32, shape=(num_classes), name="class_weights")

global_step = tf.Variable(0, name='global_step', trainable=False)

logits = segnet(X, num_classes)
predicted_image = tf.nn.softmax(logits)
# unweighted_cost = sparse_unweighted_cost(logits, sparse_label, num_classes)
# weighted_cost = sparse_weighted_cost(logits, sparse_label, class_weights, num_classes)


summ1 = tf.summary.scalar('unweighted_cost', sparse_unweighted_cost(logits, sparse_label, num_classes))
summ2 = tf.summary.scalar('weighted_cost', sparse_weighted_cost(logits, sparse_label, class_weights, num_classes))

learning_rate_node = tf.train.exponential_decay(
    learning_rate=learning_rate,
    global_step=global_step, decay_steps=1000,
    decay_rate=0.95, staircase=True)

optimizer = tf.train.GradientDescentOptimizer(
    learning_rate=learning_rate_node).minimize(sparse_weighted_cost(logits, sparse_label, class_weights, num_classes))

summ = tf.summary.merge_all()


## some debug help
print(X)
print(sparse_label)
print(logits)
print(predicted_image)
print(tf.trainable_variables())

## weights to restore should be assigned to a python variable
with tf.variable_scope('segnet', reuse=tf.AUTO_REUSE):
    b1 = tf.get_variable("conv1/conv1_1/biases")
    W1 = tf.get_variable("conv1/conv1_1/weights")
    b2 = tf.get_variable("conv1/conv1_2/biases")
    W2 = tf.get_variable("conv1/conv1_2/weights")
    b3 = tf.get_variable("conv2/conv2_1/biases")
    W3 = tf.get_variable("conv2/conv2_1/weights")
    b4 = tf.get_variable("conv2/conv2_2/biases")
    W4 = tf.get_variable("conv2/conv2_2/weights")
    b5 = tf.get_variable("conv3/conv3_1/biases")
    W5 = tf.get_variable("conv3/conv3_1/weights")
    b6 = tf.get_variable("conv3/conv3_2/biases")
    W6 = tf.get_variable("conv3/conv3_2/weights")
    b7 = tf.get_variable("conv3/conv3_3/biases")
    W7 = tf.get_variable("conv3/conv3_3/weights")
    b8 = tf.get_variable("conv4/conv4_1/biases")
    W8 = tf.get_variable("conv4/conv4_1/weights")
    b9 = tf.get_variable("conv4/conv4_2/biases")
    W9 = tf.get_variable("conv4/conv4_2/weights")
    b10 = tf.get_variable("conv4/conv4_3/biases")
    W10 = tf.get_variable("conv4/conv4_3/weights")
    b11 = tf.get_variable("conv5/conv5_1/biases")
    W11 = tf.get_variable("conv5/conv5_1/weights")
    b12 = tf.get_variable("conv5/conv5_2/biases")
    W12 = tf.get_variable("conv5/conv5_2/weights")
    b13 = tf.get_variable("conv5/conv5_3/biases")
    W13 = tf.get_variable("conv5/conv5_3/weights")


with tf.Session() as sess:
    writer = tf.summary.FileWriter('./log/summary/', sess.graph)
    restorer = tf.train.Saver({"vgg_16/conv1/conv1_1/biases": b1,
                               "vgg_16/conv1/conv1_1/weights": W1,
                               "vgg_16/conv1/conv1_2/biases": b2,
                               "vgg_16/conv1/conv1_2/weights": W2,
                               "vgg_16/conv2/conv2_1/biases": b3,
                               "vgg_16/conv2/conv2_1/weights": W3,
                               "vgg_16/conv2/conv2_2/biases": b4,
                               "vgg_16/conv2/conv2_2/weights": W4,
                               "vgg_16/conv3/conv3_1/biases": b5,
                               "vgg_16/conv3/conv3_1/weights": W5,
                               "vgg_16/conv3/conv3_2/biases": b6,
                               "vgg_16/conv3/conv3_2/weights": W6,
                               "vgg_16/conv3/conv3_3/biases": b7,
                               "vgg_16/conv3/conv3_3/weights": W7,
                               "vgg_16/conv4/conv4_1/biases": b8,
                               "vgg_16/conv4/conv4_1/weights": W8,
                               "vgg_16/conv4/conv4_2/biases": b9,
                               "vgg_16/conv4/conv4_2/weights": W9,
                               "vgg_16/conv4/conv4_3/biases": b10,
                               "vgg_16/conv4/conv4_3/weights": W10,
                               "vgg_16/conv5/conv5_1/biases": b11,
                               "vgg_16/conv5/conv5_1/weights": W11,
                               "vgg_16/conv5/conv5_2/biases": b12,
                               "vgg_16/conv5/conv5_2/weights": W12,
                               "vgg_16/conv5/conv5_3/biases": b13,
                               "vgg_16/conv5/conv5_3/weights": W13})
    sess.run(tf.global_variables_initializer())
    restorer.restore(sess, "./vgg/vgg_16.ckpt")

    saver = tf.train.Saver()

    counter = 0

    for e in range(epochs):
        start_time = time()

        for bn in range(num_batches):
            # Load batch data using data_utils
            x_train, y_train = load_data("./d/Carvana", bn)

            # Zero mean the batch
            x_train = x_train/255
            x_train = zero_mean_batch(x_train)

            # Check shapes & save images when doing test run
            if bn == 0:
                print(f"shape of x_train is = {x_train.shape}")
                print(f"shape of y_train is = {y_train.shape}")
                for name_count, image in enumerate(x_train):
                    # print(image.shape)
                    save_image(image=image, name=f"image_batch1_{name_count}.png")

            # Calculate class weights in a batch
            c_weights = batch_class_weights(y_train, num_classes)

            for i in range(len(x_train)):
                logit, image, _ = sess.run([logits, predicted_image, optimizer],
                                            feed_dict={X: np.expand_dims(x_train[i], axis=0),
                                                       sparse_label: np.expand_dims(y_train[i], axis=0),
                                                       class_weights: c_weights})

                if counter % 10 == 0:
                    summary = sess.run(summ, feed_dict={X: np.expand_dims(x_train[i], axis=0),
                                                        sparse_label: np.expand_dims(y_train[i], axis=0),
                                                        class_weights: c_weights})
                    writer.add_summary(summary, counter)

                if bn % 10 == 0:
                    # print(image.shape)
                    save_image(f"./predicted_images/image_e{e}_bn{bn}_{i}.png", image)

                counter += 1

        end_time = time()
        save_path = saver.save(sess, "./models/model.ckpt")
        duration = end_time - start_time
        print(f"epoch no. {e} : done in {duration} sec")

    writer.close()
