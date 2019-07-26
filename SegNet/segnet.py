from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from functions import maxpool_with_indices, unpool

slim = tf.contrib.slim


def segnet(inputs,
           num_classes,
           # is_training=True,
           # dropout_keep_prob=0.5,
           scope='segnet',
           weight_decay=0.0005):
    """Oxford Net VGG 16-Layers version D Example.
    Note: All the fully_connected layers have been transformed to conv2d layers.
          To use in classification mode, resize input to 224x224.
    Args:
      inputs: a tensor of size [batch_size, height, width, channels].
      num_classes: number of predicted classes. If 0 or None, the logits layer is
        omitted and the input features to the logits layer are returned instead.
      is_training: whether or not the model is being trained.
      dropout_keep_prob: the probability that activations are kept in the dropout
        layers during training.
      spatial_squeeze: whether or not should squeeze the spatial dimensions of the
        outputs. Useful to remove unnecessary dimensions for classification.
      scope: Optional scope for the variables.
      fc_conv_padding: the type of padding to use for the fully connected layer
        that is implemented as a convolutional layer. Use 'SAME' padding if you
        are applying the network in a fully convolutional manner and want to
        get a prediction map downsampled by a factor of 32 as an output.
        Otherwise, the output prediction map will be (input / 32) - 6 in case of
        'VALID' padding.
      global_pool: Optional boolean flag. If True, the input to the classification
        layer is avgpooled to size 1x1, for any input size. (This is not part
        of the original VGG architecture.)
    Returns:
      net: the output of the logits layer (if num_classes is a non-zero integer),
        or the input to the logits layer (if num_classes is 0 or None).
      end_points: a dict of tensors with intermediate activations.
    """
    with tf.variable_scope('segnet', reuse=tf.AUTO_REUSE):
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        print(f"size after conv1 is = {net.shape}")
        net, indices1 = maxpool_with_indices(net, k_size=[1, 2, 2, 1], stride=[
                                             1, 2, 2, 1], scope='pool1')
        print(f"size after pool1 is = {net.shape}")
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        print(f"size after conv2 is = {net.shape}")
        net, indices2 = maxpool_with_indices(net, k_size=[1, 2, 2, 1], stride=[
                                             1, 2, 2, 1], scope='pool2')
        print(f"size after pool2 is = {net.shape}")
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        print(f"size after conv3 is = {net.shape}")
        net, indices3 = maxpool_with_indices(net, k_size=[1, 2, 2, 1], stride=[
                                             1, 2, 2, 1], scope='pool3')
        print(f"size after pool3 is = {net.shape}")
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        print(f"size after conv4 is = {net.shape}")
        net, indices4 = maxpool_with_indices(net, k_size=[1, 2, 2, 1], stride=[
                                             1, 2, 2, 1], scope='pool4')
        print(f"size after pool4 is = {net.shape}")
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        print(f"size after conv5 is = {net.shape}")
        net, indices5 = maxpool_with_indices(net, k_size=[1, 2, 2, 1], stride=[
                                             1, 2, 2, 1], scope='pool5')
        print(f"size after pool5 is = {net.shape}")

        net = unpool(net, indices5, upsample_factor=2, scope='unpool5')
        print(f"size after unpool5 is = {net.shape}")
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], weights_regularizer=slim.l2_regularizer(
            weight_decay), scope='decoder_conv5')
        print(f"size after decoder_conv5 is = {net.shape}")
        net = unpool(net, indices4, upsample_factor=2, scope='unpool4')
        print(f"size after unpool4 is = {net.shape}")
        net = slim.repeat(net, 2, slim.conv2d, 512, [3, 3], weights_regularizer=slim.l2_regularizer(
            weight_decay), scope='decoder_conv4')
        net = slim.conv2d(net, 256, 3, scope='decoder_conv4/decoder_conv4_3')
        print(f"size after decoder_conv4 is = {net.shape}")
        net = unpool(net, indices3, upsample_factor=2, scope='unpool3')
        print(f"size after unpool3 is = {net.shape}")
        net = slim.repeat(net, 2, slim.conv2d, 256, [3, 3], weights_regularizer=slim.l2_regularizer(
            weight_decay), scope='decoder_conv3')
        net = slim.conv2d(net, 128, 3, scope='decoder_conv3/decoder_conv3_3')
        print(f"size after decoder_conv3 is = {net.shape}")
        net = unpool(net, indices2, upsample_factor=2, scope='unpool2')
        print(f"size after unpool2 is = {net.shape}")
        net = slim.conv2d(net, 128, 3, scope='decoder_conv2/decoder_conv2_1')
        net = slim.conv2d(net, 64, 3, scope='decoder_conv2/decoder_conv2_2')
        print(f"size after decoder_conv2 is = {net.shape}")
        net = unpool(net, indices1, upsample_factor=2, scope='unpool1')
        print(f"size after unpool1 is = {net.shape}")
        net = slim.conv2d(net, 64, 3, scope='decoder_conv1/decoder_conv1_1')
        print(f"size after decoder_conv1 is = {net.shape}")
        net = slim.conv2d(net, num_classes, 3,
                          activation_fn=None,
                          weights_regularizer=slim.l2_regularizer(weight_decay),
                          scope='decoder_conv1/decoder_conv1_2')
        print(f"final size after is = {net.shape}")

        return net
