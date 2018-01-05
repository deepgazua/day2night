import random

from keras import backend
from keras.callbacks import TensorBoard
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose as DeConv2D
from keras.layers import MaxPooling2D
from keras.layers import UpSampling2D as UnPooling2D

import numpy as np
import tensorflow as tf


class TensorBoardNoClosing(TensorBoard):
    def on_train_end(self, _):
        pass


class MNISTGan(object):
    def __init__(self):
        self.x_train = None
        self.x_test = None

        self.tensor_board = TensorBoardNoClosing(log_dir='logs/autoencoder')

        self.generator_autoencoder = Sequential([
            Conv2D(16, (5, 5),
                   activation='relu', padding='same', input_shape=(28, 28, 1)),     # (28, 28, 1) -> (28, 28, 16)
            MaxPooling2D((2, 2), padding='same'),                                   # (28, 28, 16) -> (14, 14, 16)
            Conv2D(16, (3, 3), activation='relu', padding='same'),                  # (14, 14, 16) -> (14, 14, 16)
            MaxPooling2D((2, 2), padding='same'),                                   # (14, 14, 16) -> (7, 7, 16)
            Conv2D(8, (3, 3), activation='relu'),                                   # (7, 7, 16) -> (5, 5, 8)

            # Assembly is the reverse process of decomposition
            DeConv2D(16, (3, 3),
                     activation='relu', name="decoder", input_shape=(5, 5, 8)),      # (5, 5, 8) -> (7, 7, 16)
            UnPooling2D((2, 2)),                                                     # (7, 7, 16) -> (14, 14, 16)
            DeConv2D(16, (3, 3), activation='relu', padding='same'),                 # (14, 14, 16) -> (14, 14, 16)
            UnPooling2D((2, 2)),                                                     # (14, 14, 16) -> (28, 28, 16)
            DeConv2D(1, (5, 5), activation='relu', padding='same')                   # (28, 28, 16) -> (28, 28, 1)
        ])

        self.generator_autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

        self.discriminator = Sequential([
            Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 2)),
            MaxPooling2D((2, 2), padding='same'),
            Conv2D(8, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2), padding='same'),
            Dense(2),
            Activation('softmax')
        ])

        self.prepare_dataset()

    def prepare_dataset(self):
        # Use channels_last data format

        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train = x_train.astype('float32') / 255.
        """:type : np.ndarray"""

        x_test = x_test.astype('float32') / 255.
        """:type : np.ndarray"""

        self.x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
        self.x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

    def fit(self):
        self.generator_autoencoder.fit(
            self.x_train, self.x_train,
            epochs=1, batch_size=128, shuffle=True,
            validation_data=(self.x_test, self.x_test),
            callbacks=[self.tensor_board]
        )

        generator_decoder = Sequential()
        for i in range(5, 10):
            generator_decoder.add(self.generator_autoencoder.layers[i])

        summary = tf.summary.image("Random Vector",
                                   generator_decoder.predict(np.random.rand(1, 5, 5, 8)))

        self.tensor_board.writer.add_summary(summary.eval(session=backend.get_session()))

        for i in range(5):
            random_test = np.reshape(random.choice(self.x_test), (1, 28, 28, 1))
            summary = tf.summary.image(
                "Auto-Encoding %d" % i,
                self.generator_autoencoder.predict(random_test)
            )
            self.tensor_board.writer.add_summary(summary.eval(session=backend.get_session()))
            self.tensor_board.writer.add_summary(tf.summary.image(
                "Auto-Encoding Original %d" % i,
                random_test
            ).eval(session=backend.get_session()))

        self.tensor_board.writer.close()

gan = MNISTGan()
gan.fit()
