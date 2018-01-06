from keras import backend
from keras.callbacks import TensorBoard
from keras.datasets import mnist
from keras.models import load_model
from keras.models import Model
from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose as DeConv2D
from keras.layers import MaxPooling2D
from keras.layers import UpSampling2D as UnPooling2D

import numpy as np
import tensorflow as tf

import os
from os import path
import shutil

import random


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
            Conv2D(8, (3, 3), activation='relu', name='encoder'),                   # (7, 7, 16) -> (5, 5, 8)

            # Assembly is the reverse process of decomposition
            DeConv2D(16, (3, 3),
                     activation='relu', name='decoder', input_shape=(5, 5, 8)),      # (5, 5, 8) -> (7, 7, 16)
            UnPooling2D((2, 2)),                                                     # (7, 7, 16) -> (14, 14, 16)
            DeConv2D(16, (3, 3), activation='relu', padding='same'),                 # (14, 14, 16) -> (14, 14, 16)
            UnPooling2D((2, 2)),                                                     # (14, 14, 16) -> (28, 28, 16)
            DeConv2D(1, (5, 5), activation='relu', padding='same')                   # (28, 28, 16) -> (28, 28, 1)
        ])

        self.generator_autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

        self.generator_encoder = None
        self.generator_decoder = None

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
            epochs=15, batch_size=128, shuffle=True,
            validation_data=(self.x_test, self.x_test),
            callbacks=[self.tensor_board]
        )
        self.separate_model()

    def load_model(self):
        self.generator_autoencoder = load_model('models/autoencoder.h5')
        self.tensor_board.set_model(self.generator_autoencoder)
        self.separate_model()

    def separate_model(self):
        self.generator_encoder = Model(
            inputs=self.generator_autoencoder.input,
            outputs=self.generator_autoencoder.get_layer('encoder').output
        )
        self.generator_decoder = Sequential()
        for i in range(5, 10):
            self.generator_decoder.add(self.generator_autoencoder.layers[i])

        self.generator_autoencoder.save('models/autoencoder.h5')

    def write_results(self):

        # Random Vector Image Creation
        for i in range(5):
            summary = tf.summary.image("Random Vector %d" % i,
                                       self.generator_decoder.predict(np.random.rand(1, 5, 5, 8)))

            self.tensor_board.writer.add_summary(summary.eval(session=backend.get_session()))

        # Auto Encoding Test
        for i in range(5):
            random_test = np.reshape(random.choice(self.x_test), (1, 28, 28, 1))
            summary = tf.summary.image(
                "Auto-Encoding %d" % i,
                self.generator_autoencoder.predict(random_test),
                family="Auto-Encoding %d" % i
            )
            self.tensor_board.writer.add_summary(summary.eval(session=backend.get_session()))
            self.tensor_board.writer.add_summary(tf.summary.image(
                "Auto-Encoding_Original %d" % i,
                random_test,
                family="Auto-Encoding %d" % i
            ).eval(session=backend.get_session()))

        # Vector Walking
        for i in range(5):
            random_image = self.generator_encoder.predict(np.reshape(random.choice(self.x_test), (1, 28, 28, 1)))
            transition_image = self.generator_encoder.predict(np.reshape(random.choice(self.x_test), (1, 28, 28, 1)))
            transition_map = transition_image - random_image

            for j in range(10):
                for x, _ in enumerate(random_image[0]):
                    for y, __ in enumerate(random_image[0][x]):
                        random_image[0][x][y] += transition_map[0][x][y] / 10

                self.tensor_board.writer.add_summary(tf.summary.image(
                    'Vector-Space-Walking_{0}_{1}/10'.format(i, j),
                    self.generator_decoder.predict(random_image),
                    family="Vector-Space-Walking_%d" % i
                ).eval(session=backend.get_session()))

        self.tensor_board.writer.close()

for file in os.listdir('./logs/autoencoder'):
    shutil.move(path.join('./logs/autoencoder', file), './logs_old')

gan = MNISTGan()

if path.exists('./models/autoencoder.h5'):
    gan.load_model()
else:
    gan.fit()

gan.write_results()
