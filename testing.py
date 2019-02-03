import tensorflow as tf
import numpy as np

# x = tf.random_normal([1, 572, 572, 1], mean=0, stddev=0.1)
# true_output = tf.random_normal([1, 388, 388, 2], mean=0, stddev=0.1)

y_train = np.load('labels.npy')
x_train = np.load('train_imgs.npy')

for i in range(len(y_train)):
    print(x_train[i].shape)
    print(y_train[i].shape)


def conv_conv_pool(input, num_filters, filter_size, pool_size, block_nos):

    with tf.variable_scope(f"block{block_nos}", reuse=tf.AUTO_REUSE):
        W1 = tf.get_variable("w_conv1", [filter_size, filter_size,
                                         input.shape[3], num_filters],
                             initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))
        W2 = tf.get_variable("w_conv2", [filter_size, filter_size,
                                         num_filters, num_filters],
                             initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))
        b1 = tf.get_variable("bias1", [num_filters],
                             initializer=tf.constant_initializer(0.01))
        b2 = tf.get_variable("bias2", [num_filters],
                             initializer=tf.constant_initializer(0.01))


        conv_block = tf.nn.conv2d(input, filter=W1,
                                  strides=[1, 1, 1, 1], padding="VALID")
        conv_block = tf.nn.bias_add(conv_block, b1)
        conv_block = tf.nn.relu(conv_block)
        conv_block = tf.nn.conv2d(conv_block, filter=W2,
                                  strides=[1, 1, 1, 1], padding="VALID")
        conv_block = tf.nn.bias_add(conv_block, b2)
        conv_block = tf.nn.relu(conv_block)

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
                                 initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))
        W_upconv = tf.get_variable("w_upconv",
                                   [upconv_scale, upconv_scale,
                                    num_filters//2, num_filters],
                                   initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))
        b1 = tf.get_variable("bias1", [num_filters],
                             initializer=tf.constant_initializer(0.01))


        conv_block = tf.nn.conv2d(input, filter=W_conv, strides=[1, 1, 1, 1], padding="VALID")
        conv_block = tf.nn.bias_add(conv_block, b1)
        conv_block = tf.nn.relu(conv_block)

        output_shape = tf.stack([conv_block.shape[0],
                                 conv_block.shape[1]*upconv_scale,
                                 conv_block.shape[2]*upconv_scale,
                                 conv_block.shape[3]//2])
        b2 = tf.get_variable("bias2", [conv_block.shape[3]//2],
                             initializer=tf.constant_initializer(0.01))

        conv_block = tf.nn.conv2d_transpose(conv_block, W_upconv,
                                            output_shape,
                                            strides=[1, 2, 2, 1],
                                            padding="VALID")
        conv_block = tf.nn.bias_add(conv_block, b2)
        conv_block = tf.nn.relu(conv_block)

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
                             initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))
        b1 = tf.get_variable("bias", [num_filters], initializer=tf.constant_initializer(0.01))

        conv = tf.nn.conv2d(concated, W1, strides=[1, 1, 1, 1],
                            padding="VALID")
        conv = tf.nn.bias_add(conv, b1)
        conv = tf.nn.relu(conv)

    return conv


def conv2d_layer(input, num_filters, filter_size, block_nos):

    with tf.variable_scope(f"block{block_nos}", reuse=tf.AUTO_REUSE):
        W = tf.get_variable("w_conv", [filter_size, filter_size,
                                       input.shape[3], num_filters],
                            initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1))
        b = tf.get_variable("bias", [num_filters], initializer=tf.constant_initializer(0.01))

        conv = tf.nn.conv2d(input, filter=W, strides=[1, 1, 1, 1],
                            padding="VALID")
        conv = tf.nn.bias_add(conv, b)
        conv = tf.nn.relu(conv)

    return conv


####################### UNet Architecture Start #######################

num_classes = 22

X = tf.placeholder(tf.float32, shape=(1, 572, 572, 1))
y = tf.placeholder(tf.float32, shape=(1, 572, 572, 22))
y_true = tf.slice(y, begin=[0, 92, 92, 0], size=[-1, 388, 388, -1])


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
Conv14 = conv2d_layer(Conv13, 64, 3, 14)
Conv15 = conv2d_layer(Conv14, num_filters=num_classes, filter_size=1, block_nos=15)

####################### UNet Architecture End #######################


def cross_entropy(softmaxed_output, correct_output):
    clip_low = 1e-10
    clip_high = 1

    pixelwise_out = tf.reshape(softmaxed_output, [-1, num_classes])
    print(f"shape of pixelwise_out = {pixelwise_out.shape}")
    pixelwise_cor = tf.reshape(correct_output, [-1, num_classes])
    print(f"shape of pixelwise_out = {pixelwise_out.shape}")

    return tf.reduce_mean(-tf.reduce_sum(
        pixelwise_cor * tf.log(
            tf.clip_by_value(pixelwise_out, clip_value_min=clip_low, clip_value_max=clip_high)),
        axis=1), name="cross_entropy")

# def calculate_cost(output_layer, correct_map):


def pixel_wise_softmax(output_map):
    with tf.name_scope("pixel_wise_softmax"):
        max_axis = tf.reduce_max(output_map, axis=3, keepdims=True)
        exponential_map = tf.exp(output_map - max_axis)
        normalize = tf.reduce_sum(exponential_map, axis=3, keepdims=True)
        return exponential_map / normalize


def cost1(net_output, true_output):
    pixelwise_output = tf.reshape(net_output, [-1, num_classes])
    pixelwise_correct = tf.reshape(true_output, [-1, num_classes])
    print(f"shape of pixelwise_output = {pixelwise_output.shape}")
    print(f"shape of pixelwise_correct = {pixelwise_correct.shape}")
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=pixelwise_output, labels=pixelwise_correct))
    print(f"shape of cost1 = {cost.shape}")
    return cost


def cost2(net_output, true_output):
    softmaxed = tf.nn.softmax(net_output, axis=3)
    cost = cross_entropy(softmaxed, true_output)
    print(f"shape of softmaxed = {softmaxed.shape}")
    print(f"shape of cost2 = {cost.shape}")
    return cost


# Test to check if the shapes are correct
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter('/UNet/log/model_graph', sess.graph)
    writer.close()

    print([X.shape, y_true.shape, Conv5.shape, Conv15.shape])
    print(tf.trainable_variables())
    print(sess.run([X, y_true,
                    Conv5, Conv15,
                    cost1(Conv15, y_true), cost2(Conv15, y_true)],
                   feed_dict={X: x_train[0], y: y_train[0]}))
