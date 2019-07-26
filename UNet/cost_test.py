import tensorflow as tf
import numpy as np
from random import randint

num_classes = 5
input = np.random.rand(1, 4, 4, num_classes)
output = np.zeros(shape=(1, 4, 4, num_classes))
sparse_output = np.zeros(shape=(1, 4, 4))

for i in range(0, 4):
    for j in range(0, 4):
        inti = randint(0, num_classes-1)
        sparse_output[0,i,j] = inti
        output[0, i, j][inti] = 1


print(input)
print(output)
print(sparse_output)

x = tf.placeholder(tf.float32, shape=(1, 4, 4, num_classes), name="myInput")
y = tf.placeholder(tf.float32, shape=(1, 4, 4, num_classes), name="myOutput")
sparse_y = tf.placeholder(tf.float32, shape=(1, 4, 4), name="my_sparseOutput")
c_weights = tf.constant([0.4,0.2,0.1,0.15,0.15])


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


# def sparse_cross_entropy(softmaxed_output, correct_output, weights, classes):
#     with tf.variable_scope("cross_entropy"):
#         clip_low = 1e-10
#         clip_high = 1
#         pixelwise_out = tf.reshape(softmaxed_output, [-1, classes])
#         pixelwise_cor = tf.reshape(correct_output, [-1])
#
#         unweighted = pixelwise_cor * tf.log(tf.clip_by_value(pixelwise_out, clip_value_min=clip_low, clip_value_max=clip_high))
#         # print(f"shape of unweighted is {unweighted.shape}"
#         weighted = tf.divide(unweighted, weights)
#         # print(f"shape of weighted is {weighted.shape}")
#         each_pixel_loss = -tf.reduce_sum(weighted, axis=1)
#         # print(f"shape of each pixels loss = {each_pixel_loss.shape}")
#         return tf.reduce_mean(each_pixel_loss, name="cross_entropy")
# def calculate_cost(output_layer, correct_map):


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


def sparse_unweighted_cost(net_output, true_output, classes):
    with tf.variable_scope("unweighted_cost"):
        pixelwise_output = tf.reshape(net_output, [-1, classes])
        pixelwise_correct = tf.reshape(true_output, [-1])
        # print(f"shape of pixelwise_output = {pixelwise_output.shape}")
        # print(f"shape of pixelwise_correct = {pixelwise_correct.shape}")
        cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=pixelwise_output, labels=pixelwise_correct))
        # print(f"shape of cost1 = {cost.shape}")
        return cost


def sparse_weighted_cost1(net_output, true_output, weights, classes):
    with tf.variable_scope("unweighted_cost"):
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


def weighted_cost(net_output, true_output, weights, classes):
    with tf.variable_scope("weighted_cost"):
        softmaxed = tf.nn.softmax(net_output, axis=3)
        cost = cross_entropy(softmaxed, true_output, weights, classes)
        # print(f"shape of softmaxed = {softmaxed.shape}")
        # print(f"shape of cost2 = {cost.shape}")
        return cost


# def sparse_weighted_cost2(net_output, true_output, weights):
#     with tf.variable_scope("weighted_cost"):
#         softmaxed = tf.nn.softmax(net_output, axis=3)
#         cost = cross_entropy(softmaxed, true_output, weights)
#         # print(f"shape of softmaxed = {softmaxed.shape}")
#         # print(f"shape of cost2 = {cost.shape}")
#         return cost

with tf.Session() as sess:
    onehot_u_cost, onehot_w_cost = sess.run([unweighted_cost(x,y,num_classes), weighted_cost(x, y, c_weights)], feed_dict={x: input,
                                                                                                                           y: output})
    print(f"onehot unweighted cost = {onehot_u_cost}")
    print(f"onehot weighted cost = {onehot_w_cost}")
