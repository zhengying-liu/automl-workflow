# -*- coding: utf-8 -*-
# @Date    : 2019/12/29 5:32
# @Author  : stellahong (stellahong@fuzhi.ai)
# @Desc    :
from keras.regularizers import L1L2, Regularizer
from keras.engine.topology import Layer
from keras.engine import InputSpec
from keras.layers import Flatten
from keras import backend as K
import tensorflow as tf


class AttentionSelf(Layer):
    """
        self attention,
        codes from:  https://mp.weixin.qq.com/s/qmJnyFMkXVjYBwoR_AQLVA
    """

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super().__init__(**kwargs)

    def build(self, input_shape):
        # W、K and V
        self.kernel = self.add_weight(name='WKV',
                                      shape=(3, input_shape[2], self.output_dim),
                                      initializer='uniform',
                                      regularizer=L1L2(0.0000032),
                                      trainable=True)
        super().build(input_shape)

    def call(self, x):
        WQ = K.dot(x, self.kernel[0])
        WK = K.dot(x, self.kernel[1])
        WV = K.dot(x, self.kernel[2])
        print("WQ.shape", WQ.shape)
        print("K.permute_dimensions(WK, [0, 2, 1]).shape", K.permute_dimensions(WK, [0, 2, 1]).shape)
        QK = K.batch_dot(WQ, K.permute_dimensions(WK, [0, 2, 1]))
        QK = QK / (64 ** 0.5)
        QK = K.softmax(QK)
        print("QK.shape", QK.shape)
        V = K.batch_dot(QK, WV)
        return V

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)


class K_Max_Pooling(Layer):
    """
    K-max pooling layer that extracts the k-highest activations from a sequence (2nd dimension).
    TensorFlow backend.
    """

    def __init__(self, k=1, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)
        self.k = k

    def compute_output_shape(self, input_shape):
        return (input_shape[0], (input_shape[2] * self.k))

    def call(self, inputs):
        # swap last two dimensions since top_k will be applied along the last dimension
        # shifted_input = tf.transpose(inputs, [0, 2, 1])

        # extract top_k, returns two tensors [values, indices]
        top_k = tf.nn.top_k(inputs, k=self.k, sorted=True, name=None)[0]

        # return flattened output
        return Flatten()(top_k)
