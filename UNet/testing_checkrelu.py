import tensorflow as tf
import numpy as np
import os
from time import time
import matplotlib.pyplot as plot

current_dir = os.getcwd()
# x = tf.random_normal([1, 572, 572, 1], mean=0, stddev=0.1)
# true_output = tf.random_normal([1, 388, 388, 2], mean=0, stddev=0.1)

y_train = np.load(f"{current_dir}/d/train_y/y_labels0.npy")
x_train = np.load(f"{current_dir}/d/train_x/x_orig0.npy")

img_count = 0
for image in y_train:
    img = np.zeros((132, 132))
    for j in range(0, 132):
        for k in range(0, 132):
            img[j, k] = (np.argmax(image[0, j, k], axis=0) * 12)
    plot.imsave(f"label{img_count}.png", img, cmap='prism')
    img_count += 1

print(f"shape of x_train is {x_train.shape}")
print(f"shape of y_train is {y_train.shape}")

mean = np.mean(x_train, axis=0)
for i in range(len(x_train)):
    x_train[i] = (x_train[i] - mean)

instances = np.zeros((22))
class_weights = []
for image in y_train:
    for i in range(132):
        for j in range(132):
            trueclass_index = np.argmax(image[0, i, j])
            instances[trueclass_index] += 1

background_ratio = instances[0]/np.sum(instances)
others_ratio = (1-background_ratio)/21
class_weights.append(background_ratio)
for n in range(1, 22):
    class_weights.append(others_ratio)
print(class_weights)

print(f"true background prediction = {np.mean(y_train[0][0, :, :, 0])}")
print(f"true boundaries prediction = {np.mean(y_train[0][0, :, :, 21])}")

num_classes = 22
learning_rate = 1e-3
epochs = 40

# for i in range(len(y_train)):
#     print(x_train[i].shape)
#     print(y_train[i].shape)

def weight_initializer(stddev):
    normal_initializer = tf.truncated_normal_initializer(mean=0, stddev=stddev)
    return normal_initializer
    # he_initializer = tf.initializers.he_normal()


def conv_conv_pool(input, num_filters, filter_size, pool_size, block_nos):

    with tf.variable_scope(f"block{block_nos}", reuse=tf.AUTO_REUSE):
        W1 = tf.get_variable("w_conv1", [filter_size, filter_size,
                                         input.shape[3], num_filters],
                             initializer=weight_initializer(np.sqrt(2 / (filter_size ** 2 * num_filters))))
        W2 = tf.get_variable("w_conv2", [filter_size, filter_size,
                                         num_filters, num_filters],
                             initializer=weight_initializer(np.sqrt(2 / (filter_size ** 2 * num_filters))))
        b1 = tf.get_variable("bias1", [num_filters],
                             initializer=tf.constant_initializer(0.1))
        b2 = tf.get_variable("bias2", [num_filters],
                             initializer=tf.constant_initializer(0.1))

        tf.summary.histogram("weights1", W1)
        tf.summary.histogram("weights2", W2)
        tf.summary.histogram("biases1", b1)
        tf.summary.histogram("biases2", b2)

        conv_block = tf.nn.conv2d(input, filter=W1,
                                  strides=[1, 1, 1, 1], padding="VALID")
        conv_block = tf.nn.bias_add(conv_block, b1)
        conv_block = tf.nn.relu(conv_block)
        conv_block = tf.nn.conv2d(conv_block, filter=W2,
                                  strides=[1, 1, 1, 1], padding="VALID")
        pre_relu = tf.nn.bias_add(conv_block, b2)
        tf.summary.histogram("pre_relu", pre_relu)

        conv_block = tf.nn.relu(pre_relu)
        tf.summary.histogram("activation", conv_block)

        pooled_block = tf.nn.max_pool(conv_block,
                                      ksize=[1, pool_size, pool_size, 1],
                                      strides=[1, pool_size, pool_size, 1],
                                      padding="VALID")


    return conv_block, pooled_block


def conv_upconv(input, num_filters, conv_filter_size, upconv_scale, block_nos):

    with tf.variable_scope(f"block{block_nos}", reuse=tf.AUTO_REUSE):
        W_conv = tf.get_variable("w_conv",
                                 [conv_filter_size, conv_filter_size,
                                  input.shape[3], num_filters],
                                 initializer=weight_initializer(np.sqrt(2 / (conv_filter_size ** 2 * num_filters))))
        W_upconv = tf.get_variable("w_upconv",
                                   [upconv_scale, upconv_scale,
                                    num_filters//2, num_filters],
                                   initializer=weight_initializer(np.sqrt(2 / (conv_filter_size ** 2 * (num_filters/2)))))
        b1 = tf.get_variable("bias1", [num_filters],
                             initializer=tf.constant_initializer(0.1))

        conv_block = tf.nn.conv2d(input, filter=W_conv, strides=[1, 1, 1, 1], padding="VALID")
        conv_block = tf.nn.bias_add(conv_block, b1)
        conv_block = tf.nn.relu(conv_block)

        output_shape = tf.stack([conv_block.shape[0],
                                 conv_block.shape[1]*upconv_scale,
                                 conv_block.shape[2]*upconv_scale,
                                 conv_block.shape[3]//2])
        b2 = tf.get_variable("bias2", [conv_block.shape[3]//2],
                             initializer=tf.constant_initializer(0.1))

        tf.summary.histogram("weights", W_conv)
        tf.summary.histogram("weights_upconv", W_upconv)
        tf.summary.histogram("biases1", b1)
        tf.summary.histogram("biases2", b2)

        conv_block = tf.nn.conv2d_transpose(conv_block, W_upconv,
                                            output_shape,
                                            strides=[1, 2, 2, 1],
                                            padding="VALID")
        conv_block = tf.nn.bias_add(conv_block, b2)
        tf.summary.histogram("pre_relu", conv_block)

        conv_block = tf.nn.relu(conv_block)
        tf.summary.histogram("activation", conv_block)

    return conv_block


def concat_conv(prev_conved, upconved, num_filters, filter_size, block_nos):

    target_height = upconved.shape[2]
    target_width = upconved.shape[1]

    with tf.variable_scope(f"block{block_nos}", reuse=tf.AUTO_REUSE):

        offsets = tf.stack([0, (prev_conved.shape[1]-target_height)//2,
                            (prev_conved.shape[1]-target_height)//2, 0])
        size_of_cropped = tf.stack([-1, target_width, target_height, -1])

        cropped = tf.slice(prev_conved, begin=offsets, size=size_of_cropped)
        concated = tf.concat(values=[cropped, upconved], axis=3)

        W1 = tf.get_variable("w_conv", [filter_size, filter_size,
                                        concated.shape[3], num_filters],
                             initializer=weight_initializer(np.sqrt(2 / (filter_size ** 2 * num_filters))))
        b1 = tf.get_variable("bias", [num_filters], initializer=tf.constant_initializer(0.1))

        tf.summary.histogram("weights", W1)
        tf.summary.histogram("biases", b1)

        conv = tf.nn.conv2d(concated, W1, strides=[1, 1, 1, 1],
                            padding="VALID")
        pre_relu = tf.nn.bias_add(conv, b1)
        tf.summary.histogram("pre_relu", pre_relu)

        conv = tf.nn.relu(pre_relu)
        tf.summary.histogram("activation", conv)

    return conv


def conv2d_layer(input, num_filters, filter_size, block_nos, last_flag=False):

    with tf.variable_scope(f"block{block_nos}", reuse=tf.AUTO_REUSE):
        W = tf.get_variable("w_conv", [filter_size, filter_size,
                                       input.shape[3], num_filters],
                            initializer=weight_initializer(np.sqrt(2 / (filter_size ** 2 * num_filters))))
        b = tf.get_variable("bias", [num_filters], initializer=tf.constant_initializer(0.1))

        tf.summary.histogram("weights", W)
        tf.summary.histogram("biases", b)

        conv = tf.nn.conv2d(input, filter=W, strides=[1, 1, 1, 1],
                            padding="VALID")
        conv = tf.nn.bias_add(conv, b)
        tf.summary.histogram("pre_relu", conv)

        if last_flag:
            return conv
        else:
            conv = tf.nn.relu(conv + b)
            tf.summary.histogram("activation", conv)

    return conv


####################### UNet Architecture Start #######################


X = tf.placeholder(tf.float32, shape=(1, 316, 316, 1), name="myInput")
y = tf.placeholder(tf.float32, shape=(1, 132, 132, 22), name="myOutput")

global_step = tf.Variable(0, "global_step")

Conv1, Pool1 = conv_conv_pool(input=X, num_filters=64, filter_size=3, pool_size=2, block_nos=1)
Conv2, Pool2 = conv_conv_pool(Pool1, 128, 3, 2, 2)
Conv3, Pool3 = conv_conv_pool(Pool2, 256, 3, 2, 3)
Conv4, Pool4 = conv_conv_pool(Pool3, 512, 3, 2, 4)
Conv5 = conv2d_layer(input=Pool4, num_filters=1024, filter_size=3, block_nos=5)
Conv6 = conv_upconv(input=Conv5, num_filters=1024, conv_filter_size=3, upconv_scale=2, block_nos=6)
Conv7 = concat_conv(prev_conved=Conv4, upconved=Conv6, num_filters=512, filter_size=3, block_nos=7)
Conv8 = conv_upconv(Conv7, 512, 3, 2, 8)
Conv9 = concat_conv(Conv3, Conv8, 256, 3, 9)
Conv10 = conv_upconv(Conv9, 256, 3, 2, 10)
Conv11 = concat_conv(Conv2, Conv10, 128, 3, 11)
Conv12 = conv_upconv(Conv11, 128, 3, 2, 12)
Conv13 = concat_conv(Conv1, Conv12, 64, 3, 13)
Conv14 = conv2d_layer(Conv13, 64, 3, 14, last_flag=True)
Conv15 = conv2d_layer(Conv14, num_filters=num_classes, filter_size=1, block_nos=15, last_flag=True)

####################### UNet Architecture End #######################


def cross_entropy(softmaxed_output, correct_output, weights):
    with tf.variable_scope("cross_entropy"):
        clip_low = 1e-10
        clip_high = 1
        pixelwise_out = tf.reshape(softmaxed_output, [-1, num_classes])
        # print(f"shape of pixelwise_out = {pixelwise_out.shape}")
        pixelwise_cor = tf.reshape(correct_output, [-1, num_classes])
        # print(f"shape of pixelwise_cor = {pixelwise_cor.shape}")

        # return tf.reduce_mean(-tf.reduce_sum(
        #     pixelwise_cor * tf.log(
        #         tf.clip_by_value(pixelwise_out, clip_value_min=clip_low, clip_value_max=clip_high)),
        #     axis=1), name="cross_entropy")

        unweighted = pixelwise_cor * tf.log(tf.clip_by_value(pixelwise_out, clip_value_min=clip_low, clip_value_max=clip_high))
        #print(f"shape of unweighted is {unweighted.shape}")
        weighted = tf.divide(unweighted, weights)
        #print(f"shape of weighted is {weighted.shape}")
        each_pixel_loss = -tf.reduce_sum(weighted, axis=1)
        #print(f"shape of each pixels loss = {each_pixel_loss.shape}")
        return tf.reduce_mean(each_pixel_loss, name="cross_entropy")

# def calculate_cost(output_layer, correct_map):


def pixel_wise_softmax(output_map):
    with tf.name_scope("pixel_wise_softmax"):
        max_axis = tf.reduce_max(output_map, axis=3, keepdims=True)
        exponential_map = tf.exp(output_map - max_axis)
        normalize = tf.reduce_sum(exponential_map, axis=3, keepdims=True)
        return exponential_map / normalize


def cost1(net_output, true_output):
    with tf.variable_scope("tf_cost"):
        pixelwise_output = tf.reshape(net_output, [-1, num_classes])
        pixelwise_correct = tf.reshape(true_output, [-1, num_classes])
        # print(f"shape of pixelwise_output = {pixelwise_output.shape}")
        # print(f"shape of pixelwise_correct = {pixelwise_correct.shape}")
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=pixelwise_output, labels=pixelwise_correct))
        # print(f"shape of cost1 = {cost.shape}")
        return cost
summ1 = tf.summary.scalar('unweighted_cost', cost1(Conv15, y))


def cost2(net_output, true_output, weights):
    with tf.variable_scope("own_cost"):
        softmaxed = tf.nn.softmax(net_output, axis=3)
        cost = cross_entropy(softmaxed, true_output, weights)
        # print(f"shape of softmaxed = {softmaxed.shape}")
        # print(f"shape of cost2 = {cost.shape}")
        return cost
summ2 = tf.summary.scalar('weighted_cost', cost2(Conv15, y, class_weights))


learning_rate_node = tf.train.exponential_decay(
    learning_rate=learning_rate,
    global_step=global_step, decay_steps=100,
    decay_rate=0.95, staircase=True)

summ = tf.summary.merge_all()

optimizer = tf.train.GradientDescentOptimizer(
    learning_rate=learning_rate_node).minimize(cost2(Conv15, y, class_weights))

# grads = optimizer.compute_gradients(loss=cost1(Conv15, y_true),
#                                     var_list=[tf.trainable_variables()])
# minimize = optimizer.apply_gradients(grads)

# Test to check if the shapes are correct
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter('/UNet/log/model_graph/', sess.graph)
    saver = tf.train.Saver()

    print([X.shape, y.shape, Conv5.shape, Conv15.shape])
    print(tf.trainable_variables())

    # to check the initializer for the first image
    # print(sess.run([X, y_true,
    #                 Conv5, Conv15,
    #                 cost1(Conv15, y_true), cost2(Conv15, y_true)],
    #                 feed_dict={X: x_train[0], y: y_train[0]}))
    counter = 0
    for e in range(epochs):
        start_time = time()
        for i in range(len(x_train)):
            logits, image, _ = sess.run([Conv15,
                                         pixel_wise_softmax(Conv15),
                                         optimizer],
                                        feed_dict={X: x_train[i],
                                                   y: y_train[i]})
            if counter%10 == 0:
                summary = sess.run(summ, feed_dict={X: x_train[i], y: y_train[i]})
                writer.add_summary(summary, counter)

            if i == 0:
                print(f"avg background prediction = {np.mean(image[0, :, :, 0])}")
                print(f"avg boundaries prediction = {np.mean(image[0, :, :, 21])}")

            img = np.zeros((132, 132))
            for j in range(0, 132):
                for k in range(0, 132):
                    img[j, k] = (np.argmax(image[0, j, k], axis=0) * 12)
            plot.imsave(f"image{i}.png", img, cmap='prism')

            counter += 1

        end_time = time()
        save_path = saver.save(sess, "./tmp/model.ckpt")
        duration = end_time - start_time
        print(f"epoch no. {e} : done in {duration} sec")

    writer.close()
