# -*- coding:utf-8 -*-

import tensorflow as tf


def fire_module(inputs, squeeze_num, expand, name, activation=tf.nn.relu):
    """
    :param inputs: Tensor
    :param squeeze_num: int
    :param expand: int
    :param name: str
    :param activation: Non Linear Activation Function
    :return: Tensor
    """

    def conv2d(input_tensor, filter_num, kernel, stride, name):
        return tf.layers.conv2d(input=input_tensor, filter=filter_num, kernel_size=(kernel, kernel),
                                stride=(stride, stride), padding='same', activation=activation, name=name)

    expand_num = expand // 2

    with tf.variable_scope(name, 'FireModule'):
        squeezed = conv2d(input_tensor=inputs, filter_num=squeeze_num, kernel=1, stride=1, name='squeezed')
        expand_1x1 = conv2d(input_tensor=squeezed, filter_num=expand_num, kernel=1, stride=1, name='1x1')
        expand_3x3 = conv2d(input_tensor=squeezed, filter_num=expand_num, kernel=3, stride=1, name='3x3')

        # channel last
        out = tf.concat([expand_1x1, expand_3x3], axis=-1)
        return out


def fire_module_with_shortcut(inputs, squeeze_num, expand, name):
    """
    :param inputs:
    :param squeeze_num:
    :param expand:
    :param name:
    :return:
    """
    with tf.variable_scope(name, 'FireModuleWithShortcut'):
        fire1 = fire_module(inputs=inputs, squeeze_num=squeeze_num, expand=expand, name='Fire1')
        fire2 = fire_module(inputs=fire1, squeeze_num=squeeze_num, expand=expand, name='Fire1')
        out = fire1 + fire2

        return out
