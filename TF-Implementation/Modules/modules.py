# -*- coding:utf-8 -*-

import tensorflow as tf


def global_average_pooling(inputs, dimension):
    """
    :param inputs: Tensor
    :param dimension: int
    :return:
    """
    for _ in range(dimension):
        inputs = tf.reduce_mean(inputs, axis=1)
    return inputs
