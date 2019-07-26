import tensorflow as tf
import numpy as np


def weight_initializer(stddev=0.1, he=False):
    if (he == True):
        return tf.initializers.he_normal()
    else:
        return tf.truncated_normal_initializer(mean=0, stddev=stddev)


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

        # tf.summary.histogram("weights1", W1)
        # tf.summary.histogram("weights2", W2)
        # tf.summary.histogram("biases1", b1)
        # tf.summary.histogram("biases2", b2)

        conv_block = tf.nn.conv2d(input, filter=W1,
                                  strides=[1, 1, 1, 1], padding="SAME")
        conv_block = tf.nn.bias_add(conv_block, b1)
        conv_block = tf.nn.relu(conv_block)

        print(conv_block.shape)

        conv_block = tf.nn.conv2d(conv_block, filter=W2,
                                  strides=[1, 1, 1, 1], padding="SAME")
        pre_relu = tf.nn.bias_add(conv_block, b2)

        print(conv_block.shape)

        # tf.summary.histogram("pre_relu", pre_relu)

        conv_block = tf.nn.relu(pre_relu)
        # tf.summary.histogram("activation", conv_block)

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

        conv_block = tf.nn.conv2d(input, filter=W_conv, strides=[1, 1, 1, 1], padding="SAME")
        conv_block = tf.nn.bias_add(conv_block, b1)
        conv_block = tf.nn.relu(conv_block)

        print(conv_block.shape)

        output_shape = tf.stack([conv_block.shape[0],
                                 conv_block.shape[1]*upconv_scale,
                                 conv_block.shape[2]*upconv_scale,
                                 conv_block.shape[3]//2])
        b2 = tf.get_variable("bias2", [conv_block.shape[3]//2],
                             initializer=tf.constant_initializer(0.1))

        # tf.summary.histogram("weights", W_conv)
        # tf.summary.histogram("weights_upconv", W_upconv)
        # tf.summary.histogram("biases1", b1)
        # tf.summary.histogram("biases2", b2)

        conv_block = tf.nn.conv2d_transpose(conv_block, W_upconv,
                                            output_shape,
                                            strides=[1, 2, 2, 1],
                                            padding="SAME")
        conv_block = tf.nn.bias_add(conv_block, b2)
        # tf.summary.histogram("pre_relu", conv_block)

        conv_block = tf.nn.relu(conv_block)
        # tf.summary.histogram("activation", conv_block)

        print(conv_block.shape)

    return conv_block


def concat_conv(prev_conved, upconved, num_filters, filter_size, block_nos):

    with tf.variable_scope(f"block{block_nos}", reuse=tf.AUTO_REUSE):

        # target_height = upconved.shape[2]
        # target_width = upconved.shape[1]
        # offsets = tf.stack([0, (prev_conved.shape[1]-target_height)//2,
        #                     (prev_conved.shape[1]-target_height)//2, 0])
        # size_of_cropped = tf.stack([-1, target_width, target_height, -1])
        #
        # cropped = tf.slice(prev_conved, begin=offsets, size=size_of_cropped)

        concated = tf.concat(values=[prev_conved, upconved], axis=3)
        print(concated.shape)

        W1 = tf.get_variable("w_conv", [filter_size, filter_size,
                                        concated.shape[3], num_filters],
                             initializer=weight_initializer(np.sqrt(2 / (filter_size ** 2 * num_filters))))
        b1 = tf.get_variable("bias", [num_filters], initializer=tf.constant_initializer(0.1))

        # tf.summary.histogram("weights", W1)
        # tf.summary.histogram("biases", b1)

        conv = tf.nn.conv2d(concated, W1, strides=[1, 1, 1, 1],
                            padding="SAME")
        pre_relu = tf.nn.bias_add(conv, b1)
        # tf.summary.histogram("pre_relu", pre_relu)

        conv = tf.nn.relu(pre_relu)
        # tf.summary.histogram("activation", conv)

        print(conv.shape)

    return conv


def conv2d(input, num_filters, filter_size, block_nos, last_flag=False):

    with tf.variable_scope(f"block{block_nos}", reuse=tf.AUTO_REUSE):
        W = tf.get_variable("w_conv", [filter_size, filter_size,
                                       input.shape[3], num_filters],
                            initializer=weight_initializer(np.sqrt(2 / (filter_size ** 2 * num_filters))))
        b = tf.get_variable("bias", [num_filters], initializer=tf.constant_initializer(0.1))

        # tf.summary.histogram("weights", W)
        # tf.summary.histogram("biases", b)

        conv = tf.nn.conv2d(input, filter=W, strides=[1, 1, 1, 1],
                            padding="SAME")
        conv = tf.nn.bias_add(conv, b)
        # tf.summary.histogram("pre_relu", conv)

        print(conv.shape)

        if last_flag:
            return conv
        else:
            conv = tf.nn.relu(conv + b)
            # tf.summary.histogram("activation", conv)

    return conv


def cross_entropy(softmaxed_output, correct_output, weights, classes):
    with tf.variable_scope("cross_entropy"):
        clip_low = 1e-10
        clip_high = 1
        pixelwise_out = tf.reshape(softmaxed_output, [-1, classes])
        # print(f"shape of pixelwise_out = {pixelwise_out.shape}")
        pixelwise_cor = tf.reshape(correct_output, [-1, classes])
        # print(f"shape of pixelwise_cor = {pixelwise_cor.shape}")

        # return tf.reduce_mean(-tf.reduce_sum(
        #     pixelwise_cor * tf.log(
        #         tf.clip_by_value(pixelwise_out, clip_value_min=clip_low, clip_value_max=clip_high)),
        #     axis=1), name="cross_entropy")

        unweighted = pixelwise_cor * tf.log(tf.clip_by_value(pixelwise_out, clip_value_min=clip_low, clip_value_max=clip_high))
        # print(f"shape of unweighted is {unweighted.shape}"
        weighted = tf.divide(unweighted, weights)
        # print(f"shape of weighted is {weighted.shape}")
        each_pixel_loss = -tf.reduce_sum(weighted, axis=1)
        # print(f"shape of each pixels loss = {each_pixel_loss.shape}")
        return tf.reduce_mean(each_pixel_loss, name="cross_entropy")


def sparse_cross_entropy(softmaxed_output, correct_output, weights, classes):
    with tf.variable_scope("sparse_cross_entropy"):
        clip_low = 1e-10
        clip_high = 1

        pixelwise_out = tf.reshape(softmaxed_output, [-1, classes])
        print(f"shape of pixelwise_out = {pixelwise_out.shape}")
        length = int(pixelwise_out.shape[0])
        pixelwise_cor = tf.reshape(correct_output, [-1])

        indices = tf.stack([tf.constant(np.arange(length)), pixelwise_cor], axis=1)
        logit = tf.gather_nd(pixelwise_out, indices)
        dim_weights = tf.gather(weights, pixelwise_cor)

        unweighted = tf.log(tf.clip_by_value(logit, clip_value_min=clip_low, clip_value_max=clip_high))
        weighted = tf.divide(unweighted, dim_weights)
        return -tf.reduce_mean(weighted)


def pixel_wise_softmax(output_map):
    with tf.name_scope("pixel_wise_softmax"):
        max_axis = tf.reduce_max(output_map, axis=3, keepdims=True)
        exponential_map = tf.exp(output_map - max_axis)
        normalize = tf.reduce_sum(exponential_map, axis=3, keepdims=True)
        return exponential_map / normalize


def unweighted_cost(net_output, true_output, classes):
    with tf.variable_scope("unweighted_cost"):
        pixelwise_output = tf.reshape(net_output, [-1, classes])
        pixelwise_correct = tf.reshape(true_output, [-1, classes])
        # print(f"shape of pixelwise_output = {pixelwise_output.shape}")
        # print(f"shape of pixelwise_correct = {pixelwise_correct.shape}")
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=pixelwise_output, labels=pixelwise_correct))
        # print(f"shape of cost1 = {cost.shape}")
        return cost


def sparse_weighted_cost(net_output, true_output, weights, num_classes):
    with tf.variable_scope("sparse_weighted_cost1"):
        pixelwise_output = tf.reshape(net_output, [-1, num_classes])
        pixelwise_correct = tf.reshape(true_output, [-1])

        weighted = tf.gather(weights, pixelwise_correct)
        # print(f"shape of pixelwise_output = {pixelwise_output.shape}")
        # print(f"shape of pixelwise_correct = {pixelwise_correct.shape}")
        # print(weighted)
        unweighted_cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pixelwise_output, labels=pixelwise_correct)
        cost = unweighted_cost/weighted
        return tf.reduce_mean(unweighted_cost), tf.reduce_mean(cost)


def weighted_cost(net_output, true_output, weights, classes):
    with tf.variable_scope("weighted_cost"):
        softmaxed = tf.nn.softmax(net_output, axis=3)
        cost = cross_entropy(softmaxed, true_output, weights, classes)
        # print(f"shape of softmaxed = {softmaxed.shape}")
        # print(f"shape of cost2 = {cost.shape}")
        return cost


def sparse_weighted_cost1(net_output, true_output, weights, classes):
    with tf.variable_scope("sparse_weighted_cost2"):
        softmaxed = tf.nn.softmax(net_output, axis=3)
        cost = sparse_cross_entropy(softmaxed, true_output, weights, classes)
        # print(f"shape of softmaxed = {softmaxed.shape}")
        # print(f"shape of cost2 = {cost.shape}")
        return cost


def tf_get_mask(batch_size, H, W, prediction, num_classes):
    colors = tf.random.uniform([num_classes, 3])
    pred_mask = tf.argmax(prediction, axis=-1)
    mask = tf.gather_nd(colors, tf.expand_dims(pred_mask, axis=-1))
    return mask
