import tensorflow as tf
import numpy as np
from functions import conv_conv_pool, conv_upconv, concat_conv, conv2d


def unet(X, classes):
    # X is a placeholder !
    Conv1, conv = conv_conv_pool(input=X, num_filters=64,
                                 filter_size=3, pool_size=2, block_nos=1)

    Conv2, conv = conv_conv_pool(conv, 128, 3, 2, 2)

    Conv3, conv = conv_conv_pool(conv, 256, 3, 2, 3)

    Conv4, conv = conv_conv_pool(conv, 512, 3, 2, 4)

    conv = conv2d(input=conv, num_filters=1024, filter_size=3, block_nos=5)

    conv = conv_upconv(input=conv, num_filters=1024,
                       conv_filter_size=3, upconv_scale=2, block_nos=6)

    conv = concat_conv(prev_conved=Conv4, upconved=conv,
                       num_filters=512, filter_size=3, block_nos=7)

    conv = conv_upconv(conv, 512, 3, 2, 8)

    conv = concat_conv(Conv3, conv, 256, 3, 9)

    conv = conv_upconv(conv, 256, 3, 2, 10)

    conv = concat_conv(Conv2, conv, 128, 3, 11)

    conv = conv_upconv(conv, 128, 3, 2, 12)

    conv = concat_conv(Conv1, conv, 64, 3, 13)

    conv = conv2d(conv, 64, 3, 14)

    conv = conv2d(conv, num_filters=classes,
                  filter_size=1, block_nos=15, last_flag=True)

    return conv
