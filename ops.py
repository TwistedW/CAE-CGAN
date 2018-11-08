#-*- coding: utf-8 -*-
"""
Most codes from https://github.com/carpedm20/DCGAN-tensorflow
"""
import math
import numpy as np 
import tensorflow as tf

from tensorflow.python.framework import ops

from utils import *

if "concat_v2" in dir(tf):
    def concat(tensors, axis, *args, **kwargs):
        return tf.concat_v2(tensors, axis, *args, **kwargs)
else:
    def concat(tensors, axis, *args, **kwargs):
        return tf.concat(tensors, axis, *args, **kwargs)

#本函数在于处理batch normalize,可精简化代码
def bn(x, is_training, scope):
    return tf.contrib.layers.batch_norm(x,
                                        decay=0.9,
                                        updates_collections=None,
                                        epsilon=1e-5,
                                        scale=True,
                                        is_training=is_training,
                                        scope=scope)

def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))

#用于经数据样本和标签拼接在一起
def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return concat([x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)

#本函数在于卷积网络的conv
def conv2d(input_, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="conv2d"):
    with tf.variable_scope(name):
        #input_.get_shape()[-1]返回输入的图像通道数，mnist的图像通道数为1，w记录的是filter
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        #此处可以参考DCGAN，filter加上步长处理
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
        #将conv加上偏值处理
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv

#本函数在于卷积网络的deconv
def deconv2d(input_, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, name="deconv2d", stddev=0.02, with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv

#本函数在激活函数的设计上有别于单纯的relu，在小于0的部分用了减小比例的处理，比较像ELU
def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)

#本函数在于对数据做形状上的改变，只是在线性条件下完成的
def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
        initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias

def MinibatchLayer(dim_b, dim_c, inputs, name):
    # input: batch_size, n_in
    # M: batch_size, dim_b, dim_c
    m = linear(inputs, dim_b * dim_c, scope=name)
    m = tf.reshape(m, [-1, dim_b, dim_c])
    # c: batch_size, batch_size, dim_b
    c = tf.abs(tf.expand_dims(m, 0) - tf.expand_dims(m, 1))
    c = tf.reduce_sum(c, reduction_indices=[3])
    c = tf.exp(-c)
    # o: batch_size, dim_b
    o = tf.reduce_mean(c, reduction_indices=[1])
    o -= 1  # to account for the zero L1 distance of each example with itself
    # result: batch_size, n_in+dim_b
    return tf.concat([o, inputs], axis=1)
