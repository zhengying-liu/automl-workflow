import os
import tensorflow as tf
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import MaxPooling2D
from tensorflow.python.keras.models import Sequential


def cnn_load_pretrained_model(input_shape, n_classes, max_layer_num=5):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(base_dir, "ckpt/ckpt01/data01.ckpt")
    model = _make_cnn_model(input_shape, 100, max_layer_num=max_layer_num)

    model.load_weights(filename)
    model.pop()
    model.pop()
    model.add(Dense(n_classes))
    model.add(Activation("softmax"))

    return model


def _make_cnn_model(self, input_shape, n_classes, max_layer_num=5):
    model = Sequential()
    min_size = min(input_shape[:2])
    optimizer = tf.keras.optimizers.SGD(decay=1e-06, momentum=0.9)

    for i in range(max_layer_num):
        if i == 0:
            model.add(Conv2D(64, 3, input_shape=input_shape, padding="same"))
        else:
            model.add(Conv2D(64, 3, padding="same"))

        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))

        min_size //= 2

        if min_size < 2:
            break

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Dropout(rate=0.5))
    model.add(Activation("relu"))
    model.add(Dense(n_classes))
    model.add(Activation("softmax"))
    model.compile(optimizer, "categorical_crossentropy")

    return model