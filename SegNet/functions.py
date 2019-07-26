from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np

slim = tf.contrib.slim

def vgg_arg_scope(weight_decay=0.0005):
  '''
  Defines the VGG arg scope.
  Args:
    weight_decay: The l2 regularization coefficient.
  Returns:
    An arg_scope.
  '''
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      biases_initializer=tf.zeros_initializer()):
    with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
        return arg_sc


def weight_initializer(stddev=0.1, he=False):
    if (he == True):
        return tf.initializers.he_normal()
    else:
        return tf.truncated_normal_initializer(mean=0, stddev=stddev)


def maxpool_with_indices(input, k_size, stride, scope):
    with tf.variable_scope(scope):
        pooled, indices = tf.nn.max_pool_with_argmax(input, k_size, stride, padding="VALID")
        return pooled, indices


def unpool(input, indices, upsample_factor, scope):
    with tf.variable_scope(scope):
        # input and indices must be of same shape
        new_size = tf.stack([input.shape[0],
                                input.shape[1]*upsample_factor,
                                input.shape[2]*upsample_factor,
                                input.shape[3]])
        batch_num = input.shape[0]
        flattened_size = input.shape[1]*upsample_factor*input.shape[2]*upsample_factor*input.shape[3]
        flat2 = input.shape[1]*input.shape[2]*input.shape[3]

        indices = tf.reshape(indices, [input.shape[0], -1])
        batch_indices = [tf.fill([flat2], i) for i in range(batch_num)]
        batch_indices = tf.dtypes.cast(tf.stack(batch_indices), tf.int64)
        # print(batch_indices)
        # print(indices)

        indices = tf.concat([tf.expand_dims(batch_indices, axis=-1), tf.expand_dims(indices, axis=-1)], axis=2)
        # print(indices)
        indices = tf.reshape(indices, [-1, 2])
        reshaped_input = tf.reshape(input, [-1])
        # print(input)

        scatter = tf.scatter_nd(indices, reshaped_input, tf.constant([input.shape[0], flattened_size], tf.int64))
        scatter = tf.reshape(scatter, new_size)
        # print(scatter)
        return scatter


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


def sparse_unweighted_cost(net_output, true_output, classes):
    with tf.variable_scope("sparse_unweighted_cost"):
        pixelwise_output = tf.reshape(net_output, [-1, classes])
        pixelwise_correct = tf.reshape(true_output, [-1])
        # print(f"shape of pixelwise_output = {pixelwise_output.shape}")
        # print(f"shape of pixelwise_correct = {pixelwise_correct.shape}")
        cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=pixelwise_output, labels=pixelwise_correct))
        # print(f"shape of cost1 = {cost.shape}")
        return cost


def sparse_weighted_cost(net_output, true_output, weights, classes):
    with tf.variable_scope("sparse_weighted_cost"):
        pixelwise_output = tf.reshape(net_output, [-1, classes])
        pixelwise_correct = tf.reshape(true_output, [-1])

        weighted = tf.gather(weights, pixelwise_correct)
        # print(f"shape of pixelwise_output = {pixelwise_output.shape}")
        # print(f"shape of pixelwise_correct = {pixelwise_correct.shape}")
        cost = tf.reduce_mean(tf.divide(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=pixelwise_output, labels=pixelwise_correct),
            weighted))
        # print(f"shape of cost1 = {cost.shape}")
        return cost
