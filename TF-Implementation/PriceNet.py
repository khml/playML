# -*- coding:utf-8 -*-

import tensorflow as tf

from Modules.FireModules import fire_module_with_shortcut
from Modules.modules import global_average_pooling


def PriceNet(inputs, is_training):
    """
    from https://arxiv.org/pdf/1803.11227.pdf
    :param inputs: Tensor
    :param is_training: Bool
    :return:
    """
    first_conv_layer_kernel_size = (1, 1)
    first_conv_layer_stride_size = (1, 1)
    last_conv_layer_kernel_size = (1, 1)
    last_conv_layer_stride_size = (1, 1)

    pooling_size = (3, 3)
    pooling_stride = (2, 2)

    def max_pooling2d(input_tensor, name):
        return tf.layers.max_pooling2d(inputs=input_tensor, pool_size=pooling_size,
                                       strides=pooling_stride, name=name)

    def batch_norm_relu(input_tensor):
        out = tf.layers.batch_normalization(inputs=input_tensor, training=is_training)
        out = tf.nn.relu(out)
        return out

    def fire_block(inputs_tensor, squeeze_num, expand, name, insert_pooling=True):
        with tf.variable_scope(name):
            out = fire_module_with_shortcut(inputs=inputs_tensor, squeeze_num=squeeze_num, expand=expand, name='fire')
            if insert_pooling:
                out = max_pooling2d(input_tensor=out, name='MaxPooling2')
            out = batch_norm_relu(out)
            return out

    with tf.variable_scope('First Conv'):
        conv1 = tf.layers.conv2d(input=inputs, filter=64, kernel_size=first_conv_layer_kernel_size,
                                 stride=first_conv_layer_stride_size, padding='same', name='conv')

        conv1 = max_pooling2d(input_tensor=conv1, name='MaxPooling1')
        conv1 = batch_norm_relu(conv1)

    fire1 = fire_block(inputs_tensor=conv1, squeeze_num=64, expand=128, name='fire1')
    fire2 = fire_block(inputs_tensor=fire1, squeeze_num=128, expand=256, name='fire2')
    fire3 = fire_block(inputs_tensor=fire2, squeeze_num=256, expand=384, name='fire3', insert_pooling=False)
    fire4 = fire_block(inputs_tensor=fire3, squeeze_num=384, expand=512, name='fire4', insert_pooling=False)

    with tf.variable_scope('Last Conv'):
        conv2 = tf.layers.conv2d(input=fire4, filter=256, kernel_size=last_conv_layer_kernel_size,
                                 stride=last_conv_layer_stride_size, padding='same', name='conv')
        conv2 = global_average_pooling(inputs=conv2, dimension=2)

    out = tf.layers.dense(inputs=conv2, units=1, activation=tf.nn.relu, use_bias=True)
    return out
