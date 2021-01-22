"""
MIT License

Copyright (c) 2019 Lenovo Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import os
import keras
from keras.layers import Input, LSTM, Dense, GRU, SpatialDropout1D
from keras.layers import Add, Dropout, Reshape, Concatenate, Lambda
from keras.layers import Embedding, Flatten, Conv1D, concatenate, Conv2D
from keras.layers import SeparableConv1D, BatchNormalization
from keras.layers import LeakyReLU, PReLU, ReLU, Activation
from keras.layers import MaxPooling1D, MaxPooling2D, AveragePooling1D, AveragePooling2D
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers import AlphaDropout
from keras import regularizers
from keras.regularizers import l2
from keras import backend as K
from sklearn.svm import LinearSVC
# from thundersvm import *
from sklearn.calibration import CalibratedClassifierCV
import numpy as np
from keras.layers import CuDNNGRU
from keras.models import *
from keras.layers import Bidirectional
from keras.optimizers import SGD, RMSprop, Adamax, Adadelta

import string
from keras_radam import RAdam

from Auto_NLP.upwind_flys_update.modules import AttentionSelf
from Auto_NLP.upwind_flys_update.modules import K_Max_Pooling

EMBEDDING_DIM = 300
MAX_VOCAB_SIZE = 35000

import tensorflow as tf
from tensorflow.keras.layers import Activation
from tensorflow.keras.utils import get_custom_objects

class Mish(Activation):
    '''
    Mish Activation Function.
    .. math::
        mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
    Shape:
        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
        - Output: Same shape as the input.
    Examples:
        >>> X = Activation('Mish', name="conv1_act")(X_input)
    '''

    def __init__(self, activation, **kwargs):
        super(Mish, self).__init__(activation, **kwargs)
        self.__name__ = 'Mish'


def mish(inputs):
    return inputs * tf.math.tanh(tf.math.softplus(inputs))

get_custom_objects().update({'Mish': Mish(mish)})

class ModelGenerator(object):
    def __init__(self,
                 feature_mode,
                 load_pretrain_emb=False,
                 data_feature=None,
                 meta_data_feature=None,
                 fasttext_embeddings_index=None):

        self.cnn_model_lib = {'text_cnn': ModelGenerator.text_cnn_model,
                              'text_dcnn': ModelGenerator.text_dynamic_cnn_model,
                              'text_rcnn': ModelGenerator.text_rcnn_model,
                              'text_rnn':ModelGenerator.RNN_Model,
                              'text_rcnn_pooling': ModelGenerator.text_rcnn_pooling_model,
                              'sep_cnn': ModelGenerator.sep_cnn_model,
                              'lstm_model': ModelGenerator.lstm_model,
                              'text_dpcnn': ModelGenerator.dpcnn_model,
                              'text_attn': ModelGenerator.self_attn_model,
                              # 'text_cnn_tfidf': ModelGenerator.text_cnn_wide_tfidf_model,
                              }

        self.data_feature = data_feature
        self.load_pretrain_emb = load_pretrain_emb
        self.meta_data_feature = meta_data_feature
        self.oov_cnt = 0
        # self.model_name = model_name

        if data_feature is not None:
            self.num_features = data_feature['num_features']
            self.word_index = data_feature['word_index']
            self.num_class = data_feature['num_class']
            self.max_length = data_feature['max_length']
            self.input_shape = data_feature['input_shape']

        self.feature_mode = feature_mode
        self.embedding_matrix = None
        self.use_bpe = False
        self.bpe_encoder = None
        self.lr = 0.003

        if self.feature_mode == 0:
            self.load_pretrain_emb = False
        self.fasttext_embeddings_index = fasttext_embeddings_index

    def build_model(self, model_name, data_feature):
        if model_name == 'svm':
            model = LinearSVC(random_state=0, tol=1e-5, max_iter=500)
            # model = SVC(verbose=True, gamma=0.5, C=100)
            # self.model = model
            self.model = CalibratedClassifierCV(model)
            # self.model_name = 'svm'
        else:
            self.num_features = data_feature['num_features']
            if "word_index" in data_feature:
                self.word_index = data_feature['word_index']
            self.num_class = data_feature['num_class']
            self.max_length = data_feature['max_length']
            self.input_shape = data_feature['input_shape']

            print("load_pretrain_emb", self.load_pretrain_emb)
            if self.use_bpe:
                self.generate_bpe_emb_matrix()
                self.oov_cnt = 0
            elif self.load_pretrain_emb:
                self.generate_emb_matrix()
            else:
                self.embedding_matrix = None

            kwargs = {'embedding_matrix': self.embedding_matrix,
                      'input_shape': data_feature['input_shape'],  # data_feature['input_shape'],
                      'max_length': data_feature['max_length'],
                      # todo: fasttext use num_features
                      'num_features': data_feature['num_features'], #self.embedding_matrix.shape[0]
                      'num_classes': data_feature['num_class'],
                      "filter_num": data_feature["filter_num"],
                      "trainable": False,
                      "emb_size":300
                      # "rnn_units":data_feature["rnn_units"]
                      }
            if self.model_name == "text_rcnn":
                kwargs["rnn_units"] = data_feature["rnn_units"]
            # self.model_name = 'text_cnn'

            if self.oov_cnt > 3000:  #
                self.lr = 0.001
                if model_name=="text_rcnn":
                    self.lr = 0.025
                kwargs["trainable"] = True
            else:
                if model_name == "text_cnn":
                    self.lr = 0.001
                    # kwargs["trainable"] = True
                elif model_name == "text_dpcnn":
                    self.lr = 0.016

                else:
                    if model_name == 'text_rcnn':
                        self.lr = 0.025
                        kwargs["filter_num"] = 64
                        kwargs["trainable"] = False
                    else:
                        self.lr = 0.008
            print("lr is {}".format(self.lr))

            self.model = self.cnn_model_lib[model_name](**kwargs)
            if model_name == "text_rcnn":
                opt = RAdam(learning_rate=self.lr)
            else:
                opt = keras.optimizers.RMSprop(lr=0.001)

            self.model.compile(loss="categorical_crossentropy",
                               optimizer=opt,
                               metrics=["accuracy"])

            if self.model_name not in self.cnn_model_lib.keys():
                raise Exception('incorrect model name')
        return self.model

    def load_pretrain_model(self):
        # load json and create model
        json_file = open(os.path.join(os.path.dirname(__file__), 'model.json'), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(os.path.join(os.path.dirname(__file__), "model.h5"))
        print("Loaded model from disk")
        return loaded_model

    def model_pre_select(self, call_num, snoop_avg_text_length=500, cnn_done_status=False):
        if call_num == 0:
            self.model_name = 'svm'
        elif call_num == 1:
            if snoop_avg_text_length > 300:
                self.model_name = 'text_cnn'
            else:
                self.model_name = 'text_cnn'

            print("model_name:{} \n".format(self.model_name))
            if self.model_name not in self.cnn_model_lib.keys():
                raise Exception('incorrect model name')
        elif cnn_done_status:
            self.model_name = "text_rcnn"
            print("model_name:{} \n".format(self.model_name))
            if self.model_name not in self.cnn_model_lib.keys():
                raise Exception('incorrect model name')

        print("model_name:{} \n".format(self.model_name))
        return self.model_name

    def generate_bpe_emb_matrix(self):
        print("Use bpe_emb!!!\n\n")
        pad_emb = np.random.uniform(
            -0.05, 0.05, size=300).reshape(-1, 300)
        self.embedding_matrix = np.concatenate([self.bpe_encoder.vectors, pad_emb], axis=0)
        # self.embedding_matrix = self.bpe_encoder.vectors

    def convert_emb_to_numpy(self, vocab_size, embed_size=300):
        # feat_size = len(self.embedding_matrix[list(self.embedding_matrix.keys())[0]])
        embed = np.float32(
            np.random.uniform(-0.2, 0.2, [vocab_size, embed_size]))
        for k in self.embedding_matrix.keys():
            print(k)
            embed[k] = np.array(self.embedding_matrix[k])
        return embed

    def generate_emb_matrix(self):
        cnt = 0
        self.embedding_matrix = np.zeros((self.num_features, EMBEDDING_DIM))
        for word, i in self.word_index.items():
            if i >= self.num_features:
                continue
            # word = word.translate(trantab)
            embedding_vector = self.fasttext_embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                self.embedding_matrix[i] = embedding_vector
            else:
                # self.embedding_matrix[i] = np.zeros(300)
                self.embedding_matrix[i] = np.random.uniform(
                    -0.02, 0.02, size=EMBEDDING_DIM)
                cnt += 1
        print("check self embedding_vector ", self.embedding_matrix.shape)
        # emb = np.array()
        self.oov_cnt = cnt

        print('fastText oov words: %s' % cnt)

    @staticmethod
    def _get_last_layer_units_and_activation(num_classes):
        """Gets the # units and activation function for the last network layer.

        Args:
            num_classes: Number of classes.

        Returns:
            units, activation values.
        """
        activation = 'softmax'
        units = num_classes
        return units, activation

    @staticmethod
    def text_cnn_model(input_shape,
                       embedding_matrix,
                       max_length,
                       num_features,
                       num_classes,
                       input_tensor=None,
                       filter_num=64,
                       emb_size=300,
                       trainable=False
                       ):

        inputs = Input(name='inputs', shape=[max_length], tensor=input_tensor)
        if embedding_matrix is None:
            layer = Embedding(input_dim=num_features,
                              output_dim=emb_size,
                              input_length=input_shape)(inputs)
        else:
            # num_features = MAX_VOCAB_SIZE
            layer = Embedding(input_dim=num_features,
                              output_dim=emb_size,
                              input_length=input_shape,
                              weights=[embedding_matrix],
                              # embeddings_initializer=keras.initializers.Constant(
                              #     embedding_matrix),
                              trainable=trainable)(inputs)

        cnns = []
        filter_sizes = [2, 3, 4, 5]
        conv_1_prelu = None
        for size in filter_sizes:
            cnn_l = Conv1D(filter_num,
                           size,
                           padding='same',
                           strides=1,
                           # kernel_initializer='glorot_uniform',
                           activation='relu')(layer)


            pooling_0 = MaxPooling1D(max_length - size + 1)(cnn_l)
            pooling_0 = Flatten()(pooling_0)
            cnns.append(pooling_0)

        cnn_merge = concatenate(cnns, axis=-1)
        print("check conv", cnn_merge.get_shape())
        out = Dropout(0.2)(cnn_merge)
        # out = Dense(256, activation='relu')(out)
        main_output = Dense(num_classes, activation='softmax')(out)
        model = keras.models.Model(inputs=inputs, outputs=main_output)
        return model



    @staticmethod
    def text_rcnn_pooling_model(input_shape,
                                embedding_matrix,
                                max_length,
                                num_features,
                                num_classes,
                                input_tensor=None,
                                emb_size=300,
                                filter_num=64,
                                rnn_units=128):
        inputs = Input(name='inputs', shape=[max_length], tensor=input_tensor)
        if embedding_matrix is None:
            layer = Embedding(input_dim=num_features,
                              output_dim=emb_size,
                              input_length=input_shape)(inputs)
        else:
            num_features = MAX_VOCAB_SIZE
            layer = Embedding(input_dim=num_features,
                              output_dim=emb_size,
                              input_length=input_shape,
                              embeddings_initializer=keras.initializers.Constant(
                                  embedding_matrix),
                              trainable=False)(inputs)

        # layer_cell = GRU
        layer_cell = CuDNNGRU  # 更快
        embedding_output = layer
        # 反向
        x_backwords = layer_cell(units=rnn_units,
                                 return_sequences=True,
                                 go_backwards=True)(embedding_output)
        x_backwords_reverse = Lambda(lambda x: K.reverse(x, axes=1))(x_backwords)
        # 前向
        x_fordwords = layer_cell(units=rnn_units,
                                 return_sequences=True,
                                 go_backwards=False)(embedding_output)

        # 拼接
        x_feb = Concatenate(axis=2)([x_fordwords, embedding_output, x_backwords_reverse])
        x_feb = Dropout(rate=0.35)(x_feb)
        dim_2 = K.int_shape(x_feb)[2]
        output = Dense(units=dim_2, activation='tanh')(x_feb)
        output = MaxPooling1D()(output)
        print("check output", output.get_shape())
        output = Flatten()(output)
        print("check Flatten", output.get_shape())
        output = Dense(units=num_classes, activation='softmax')(output)
        model = keras.models.Model(inputs=inputs, outputs=output)
        return model

    @staticmethod
    def text_rcnn_model(input_shape,
                        embedding_matrix,
                        max_length,
                        num_features,
                        num_classes,
                        input_tensor=None,
                        emb_size=300,
                        filter_num=64,
                        rnn_units=128,
                        trainable=False):

        inputs = Input(name='inputs', shape=[max_length], tensor=input_tensor)
        if embedding_matrix is None:
            layer = Embedding(input_dim=num_features,
                              output_dim=emb_size,
                              input_length=input_shape)(inputs)
        else:
            # num_features = MAX_VOCAB_SIZE

            num_features = embedding_matrix.shape[0]
            layer = Embedding(input_dim=num_features,
                              output_dim=emb_size,
                              # input_length=input_shape,
                              weights=[embedding_matrix],
                              # embeddings_initializer=keras.initializers.Constant(
                              #     embedding_matrix),
                              trainable=trainable)(inputs)

        # layer_cell = GRU
        layer_cell = CuDNNGRU  # 更快
        embedding_output = layer
        # 拼接
        x_feb = Bidirectional(layer_cell(units=rnn_units,
                                         return_sequences=True))(embedding_output)

        x_feb = Concatenate(axis=2)([x_feb, embedding_output])
        # x_feb = Concatenate(axis=2)([x_fordwords, embedding_output, x_backwords_reverse])

        ####使用多个卷积核##################################################
        x_feb = Dropout(rate=0.5)(x_feb)
        # Concatenate后的embedding_size
        dim_2 = K.int_shape(x_feb)[2]
        # print("check dim2", x_feb.get_shape())
        len_max = max_length
        x_feb_reshape = Reshape((len_max, dim_2, 1))(x_feb)
        # 提取n-gram特征和最大池化， 一般不用平均池化
        conv_pools = []
        filters = [2, 3, 4, 5]

        for filter_size in filters:
            conv = Conv2D(filters=filter_num,
                          kernel_size=(filter_size, dim_2),
                          padding='valid',
                          kernel_initializer='normal',
                          activation='relu',
                          )(x_feb_reshape)

            print("check conv", conv.get_shape())
            pooled = MaxPooling2D(pool_size=(len_max - filter_size + 1, 1),
                                  strides=(1, 1),
                                  padding='valid',
                                  )(conv)
            print("check pooled", pooled.get_shape())
            conv_pools.append(pooled)

        # 拼接
        x = Concatenate()(conv_pools)
        # x = concatenate(conv_pools, axis=-1)
        print("check concatenate x", x.get_shape())

        x = Flatten()(x)
        #########################################################################

        output = Dense(units=num_classes, activation='softmax')(x)
        model = keras.models.Model(inputs=inputs, outputs=output)
        return model

    @staticmethod
    def text_dynamic_cnn_model(input_shape,
                               embedding_matrix,
                               max_length,
                               num_features,
                               num_classes,
                               input_tensor=None,
                               filters=64,
                               emb_size=300,
                               emb_trainable=True,
                               pretrain_embed=None
                               ):

        inputs = Input(name='inputs', shape=[max_length], tensor=input_tensor)
        layer = pretrain_embed.embed_model.output
        # if embedding_matrix is None:
        #     layer = Embedding(input_dim=num_features,
        #                       output_dim=emb_size,
        #                       input_length=input_shape)(inputs)
        # else:
        #     num_features = MAX_VOCAB_SIZE
        #     if emb_trainable:
        #         layer = Embedding(input_dim=num_features,
        #                           output_dim=emb_size,
        #                           input_length=input_shape,
        #                           embeddings_initializer=keras.initializers.Constant(
        #                               embedding_matrix),
        #                           trainable=emb_trainable)(inputs)
        #     else:
        #         layer = Embedding(input_dim=num_features,
        #                           output_dim=emb_size,
        #                           input_length=input_shape,
        #                           embeddings_initializer=keras.initializers.Constant(
        #                               embedding_matrix))(inputs)

        cnns = []
        filter_sizes = [2, 3, 4, 5, 7]
        for size in filter_sizes:
            cnn_l = Conv1D(filters,
                           size,
                           padding='same',
                           strides=1,
                           activation='relu')(layer)
            pooling_l = MaxPooling1D(max_length - size + 1)(cnn_l)
            pooling_l = Flatten()(pooling_l)
            cnns.append(pooling_l)

        dilation_rates = [1, 2, 3]
        for size in filter_sizes:
            for dilation_rate in dilation_rates:
                d_cnn_l = Conv1D(filters,
                                 size,
                                 padding='same',
                                 strides=1,
                                 activation='relu',
                                 dilation_rate=dilation_rate)(layer)

                d_pooling_l = MaxPooling1D(max_length - size + 1)(d_cnn_l)
                d_pooling_l = Flatten()(d_pooling_l)
                cnns.append(d_pooling_l)

        cnn_merge = concatenate(cnns, axis=-1)
        out = Dropout(0.5)(cnn_merge)
        main_output = Dense(num_classes, activation='softmax')(out)
        model = keras.models.Model(inputs=inputs, outputs=main_output)
        return model

    @staticmethod
    def sep_cnn_model(input_shape,
                      max_length,
                      num_classes,
                      num_features,
                      embedding_matrix,
                      input_tensor=None,
                      emb_size=300,
                      blocks=1,
                      filters=64,
                      kernel_size=4,
                      dropout_rate=0.25):
        op_units, op_activation = ModelGenerator._get_last_layer_units_and_activation(num_classes)

        inputs = Input(name='inputs', shape=[max_length], tensor=input_tensor)
        if embedding_matrix is None:
            layer = Embedding(input_dim=num_features,
                              output_dim=emb_size,
                              input_length=input_shape)(inputs)
        else:
            num_features = MAX_VOCAB_SIZE
            layer = Embedding(input_dim=num_features,
                              output_dim=emb_size,
                              input_length=input_shape,
                              embeddings_initializer=keras.initializers.Constant(
                                  embedding_matrix))(inputs)

        for _ in range(blocks - 1):
            layer = Dropout(rate=dropout_rate)(layer)
            layer = SeparableConv1D(filters=filters,
                                    kernel_size=kernel_size,
                                    activation='relu',
                                    bias_initializer='random_uniform',
                                    depthwise_initializer='random_uniform',
                                    padding='same')(layer)
            layer = SeparableConv1D(filters=filters,
                                    kernel_size=kernel_size,
                                    activation='relu',
                                    bias_initializer='random_uniform',
                                    depthwise_initializer='random_uniform',
                                    padding='same')(layer)
            layer = MaxPooling1D(pool_size=3)(layer)

        layer = SeparableConv1D(filters=filters * 2,
                                kernel_size=kernel_size,
                                activation='relu',
                                bias_initializer='random_uniform',
                                depthwise_initializer='random_uniform',
                                padding='same')(layer)
        layer = SeparableConv1D(filters=filters * 2,
                                kernel_size=kernel_size,
                                activation='relu',
                                bias_initializer='random_uniform',
                                depthwise_initializer='random_uniform',
                                padding='same')(layer)

        layer = GlobalAveragePooling1D()(layer)
        # model.add(MaxPooling1D())
        layer = Dropout(rate=0.5)(layer)
        layer = Dense(op_units, activation=op_activation)(layer)
        model = keras.models.Model(inputs=inputs, outputs=layer)
        return model

    @staticmethod
    def lstm_model(max_length,
                   num_classes,
                   num_features,
                   embedding_matrix=None,
                   hidden_state_size=128,
                   fc1_size=256,
                   dropout_rate=0.15):
        inputs = Input(name='inputs', shape=[max_length])
        layer = Embedding(num_features, hidden_state_size,
                          input_length=max_length)(inputs)
        # layer = LSTM(hidden_state_size, return_sequences=True)(layer)
        layer = LSTM(hidden_state_size)(layer)
        layer = Dense(fc1_size, activation="relu", name="FC1")(layer)
        layer = Dropout(dropout_rate)(layer)
        layer = Dense(num_classes, activation="softmax", name="FC2")(layer)
        model = keras.models.Model(inputs=inputs, outputs=layer)
        return model

    @staticmethod
    def RNN_Model(max_length, num_classes, num_features, embedding_matrix=None):

        in_text = Input(shape=(max_length,))
        op_units, op_activation = ModelGenerator._get_last_layer_units_and_activation(num_classes)

        trainable = True
        if embedding_matrix is None:
            x = Embedding(num_features, 64, trainable=trainable)(in_text)
        else:
            x = Embedding(num_features, 300, trainable=trainable, weights=[embedding_matrix])(in_text)

        x = CuDNNGRU(128, return_sequences=True)(x)
        x = GlobalMaxPooling1D()(x)

        x = Dense(128)(x)  #
        x = PReLU()(x)
        x = Dropout(0.35)(x)  # 0
        x = BatchNormalization()(x)

        y = Dense(op_units, activation=op_activation)(x)

        md = keras.models.Model(inputs=[in_text], outputs=y)

        return md

    @staticmethod
    def CNN_Model(max_length, num_classes, num_features, embedding_matrix=None):

        in_text = Input(shape=(max_length,))
        op_units, op_activation = ModelGenerator._get_last_layer_units_and_activation(num_classes)

        trainable = True
        if embedding_matrix is None:
            x = Embedding(num_features, 64, trainable=trainable)(in_text)
        else:
            x = Embedding(num_features, 300, trainable=trainable, weights=[embedding_matrix])(in_text)

        x = Conv1D(128, kernel_size=5, padding='valid', kernel_initializer='glorot_uniform')(x)
        x = GlobalMaxPooling1D()(x)

        x = Dense(128)(x)  #
        x = PReLU()(x)
        x = Dropout(0.35)(x)  # 0
        x = BatchNormalization()(x)

        y = Dense(op_units, activation=op_activation)(x)

        md = keras.models.Model(inputs=[in_text], outputs=y)

        return md

    @staticmethod
    def ResCNN(x, filters_num, l2_ratio, activation):
        """
            repeat of two conv
        :param x: tensor, input shape
        :return: tensor, result of two conv of resnet
        """
        # pre-activation
        # x = PReLU()(x)
        x = Conv1D(filters_num,
                   kernel_size=1,
                   padding='SAME',
                   kernel_regularizer=l2(l2_ratio),
                   bias_regularizer=l2(l2_ratio),
                   activation=activation,
                   )(x)
        x = BatchNormalization()(x)
        # x = PReLU()(x)
        x = Conv1D(filters_num,
                   kernel_size=1,
                   padding='SAME',
                   kernel_regularizer=l2(l2_ratio),
                   bias_regularizer=l2(l2_ratio),
                   activation=activation,
                   )(x)
        x = BatchNormalization()(x)
        # x = Dropout(self.dropout)(x)
        x = PReLU()(x)
        return x

    @staticmethod
    def dpcnn_model(input_shape,
                    embedding_matrix,
                    max_length,
                    num_features,
                    num_classes,
                    input_tensor=None,
                    filter_num=64,
                    emb_size=300,
                    trainable=False):

        inputs = Input(name='inputs', shape=[max_length], tensor=input_tensor)
        if embedding_matrix is None:
            layer = Embedding(input_dim=num_features,
                              output_dim=emb_size,
                              input_length=input_shape)(inputs)
        else:
            # num_features = MAX_VOCAB_SIZE
            layer = Embedding(input_dim=num_features,
                              output_dim=emb_size,
                              input_length=input_shape,
                              embeddings_initializer=keras.initializers.Constant(
                                  embedding_matrix))(inputs)

        # embedding_output_spatial = SpatialDropout1D(0.2)(layer)
        l2_ratio = 0.0000032
        conv_1 = Conv1D(filter_num,
                        kernel_size=3,
                        padding='SAME',
                        kernel_regularizer=l2(l2_ratio),
                        bias_regularizer=l2(l2_ratio),
                        activation='relu',
                        )(layer)
        conv_1_prelu = PReLU()(conv_1)
        # conv_1_prelu = conv_1
        block = None
        layer_curr = 0
        layer_repeats = 2
        full_connect_unit = 256
        pooling_size_strides = [3, 2]
        for i in range(layer_repeats):
            if i == 0:  # 第一层输入用embedding输出的结果作为输入
                block = ModelGenerator.ResCNN(layer, filter_num, l2_ratio, "relu")
                block_add = Add()([block, conv_1_prelu])
                block = MaxPooling1D(pool_size=pooling_size_strides[0],
                                     strides=pooling_size_strides[1])(block_add)
            elif layer_repeats - 1 == i or layer_curr == 1:  # 最后一次repeat用GlobalMaxPooling1D
                block_last = ModelGenerator.ResCNN(block, filter_num, l2_ratio, "relu")
                # ResNet(shortcut连接|skip连接|residual连接), 这里是shortcut连接. 恒等映射, block+f(block)
                block_add = Add()([block_last, block])
                block = GlobalMaxPooling1D()(block_add)
                break
            else:  # 中间层 repeat
                if K.int_shape(block)[1] // 2 < 8:  # 防止错误, 不能pooling/2的情况, 就是说size >= 2
                    layer_curr = 1
                block_mid = ModelGenerator.ResCNN(block, filter_num, l2_ratio, "relu")
                block_add = Add()([block_mid, block])
                block = MaxPooling1D(pool_size=pooling_size_strides[0],
                                     strides=pooling_size_strides[1])(block_add)

        # 全连接层
        output = Dense(full_connect_unit, activation='relu')(block)
        output = BatchNormalization()(output)
        # output = PReLU()(output)
        output = Dropout(0.5)(output)
        main_output = Dense(num_classes, activation='softmax')(output)
        model = keras.models.Model(inputs=inputs, outputs=main_output)
        return model

    @staticmethod
    def self_attn_model(input_shape,
                        embedding_matrix,
                        max_length,
                        num_features,
                        num_classes,
                        input_tensor=None,
                        filters=64,
                        emb_size=300,
                        pretrain_embed=None):
        inputs = Input(name='inputs', shape=[max_length], tensor=input_tensor)

        if embedding_matrix is None:
            layer = Embedding(input_dim=num_features,
                              output_dim=emb_size,
                              input_length=input_shape)(inputs)
        else:
            num_features = MAX_VOCAB_SIZE
            layer = Embedding(input_dim=num_features,
                              output_dim=emb_size,
                              input_length=input_shape,
                              embeddings_initializer=keras.initializers.Constant(
                                  embedding_matrix),
                              trainable=True)(inputs)

        x = SpatialDropout1D(0.2)(layer)
        x = AttentionSelf(emb_size)(x)
        x = GlobalMaxPooling1D()(x)
        x = Dropout(0.6)(x)
        main_output = Dense(num_classes, activation='softmax')(x)
        model = keras.models.Model(inputs=inputs, outputs=main_output)
        return model

    @staticmethod
    def char_cnn_model(input_shape,
                       embedding_matrix,
                       max_length,
                       num_features,
                       num_classes,
                       input_tensor=None,
                       filter_num=64,
                       emb_size=300,
                       trainable=False):
        pass
