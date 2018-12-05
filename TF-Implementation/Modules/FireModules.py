# -*- coding:utf-8 -*-

import tensorflow as tf


def fire_module(inputs, squeeze_num, expand, name):
    """
    :param inputs: Tensor
    :param squeeze_num: int
    :param expand: int
    :param name: str
    :return: Tensor
    """
    def conv2d(input_tensor, filter_num, kernel, stride, name):
        return tf.layers.conv2d(input=input_tensor, filter=filter_num, kernel_size=(kernel, kernel),
                                stride=(stride, stride), padding='same', name=name)

    expand_num = expand // 2

    with tf.variable_scope(name, 'FireModule'):
        squeezed = conv2d(input_tensor=inputs, filter_num=squeeze_num, kernel=1, stride=1, name='squeezed')
        expand_1x1 = conv2d(input_tensor=squeezed, filter_num=expand_num, kernel=1, stride=1, name='1x1')
        expand_3x3 = conv2d(input_tensor=squeezed, filter_num=expand_num, kernel=3, stride=1, name='3x3')

        # channel last
        out = tf.concat([expand_1x1, expand_3x3], axis=-1)
        return out
