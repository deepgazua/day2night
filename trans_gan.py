from keras import backend
from keras.callbacks import TensorBoard
from keras.models import load_model
from keras.models import Model
from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose as DeConv2D
from keras.layers import MaxPooling2D
from keras.layers import UpSampling2D as UnPooling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.generic_utils import Progbar

import cv2
import numpy as np
import tensorflow as tf

import math
import os
from os import path
import random
import shutil
import time


TYPE_TRAIN = 0
TYPE_TEST = 1


def print_section(section_name):
    print("====================%s====================" % section_name)


# noinspection PyShadowingNames
def load_dataset_flow(dataset, set_type, batch_size):
    set_dir = path.join("dataset", dataset, set_type)
    data_generator = ImageDataGenerator(samplewise_std_normalization=True)
    return data_generator.flow_from_directory(
        set_dir, target_size=(256, 256), class_mode="input",
        batch_size=batch_size, shuffle=True,
    ), len(os.listdir(path.join(set_dir, "images")))


def check_directory_and_create(directory_name):
    if not path.exists(directory_name):
        os.makedirs(directory_name)

    elif path.isfile(directory_name):
        print("%s is not a directory! Aborting..." % directory_name)
        exit(1)


class TensorBoardNoClosing(TensorBoard):
    def on_train_end(self, _):
        pass

    def finish(self):
        self.writer.flush()
        self.writer.close()


class AutoEncoder(object):
    # noinspection PyShadowingNames
    def __init__(self, name, encoder_tensor_board, batch_size):
        self.name = name

        self.train = None
        self.train_count = 0

        self.test = None
        self.test_count = 0
        self.test_subset = None

        self.batch_size = batch_size

        self.tensor_board = encoder_tensor_board

        self.generator_autoencoder = Sequential([
            Conv2D(16, (5, 5),
                   activation='relu', padding='same', input_shape=(256, 256, 3)),   # (256, 256, 3) -> (256, 256, 16)
            MaxPooling2D((2, 2), padding='same'),                                   # (256, 256, 16) -> (128, 128, 16)
            Conv2D(16, (3, 3), activation='relu', padding='same'),                  # (128 128, 16) -> (128, 128, 16)
            MaxPooling2D((2, 2), padding='same'),                                   # (64, 64, 16) -> (64, 64, 16)
            Conv2D(8, (3, 3), activation='relu'),                                   # (64, 64, 16) -> (64, 64, 8)
            MaxPooling2D((2, 2), padding='same'),                                   # (64, 64, 8) -> (32, 32, 8)
            Conv2D(8, (3, 3), activation='tanh', name='encoder'),                   # (32, 32, 16) -> (32, 32, 8)

            # Assembly is the reverse process of decomposition
            DeConv2D(8, (3, 3),
                     activation='relu', name='decoder', input_shape=(16, 16, 8)),    # (32, 32, 8) -> (32, 32, 8)
            UnPooling2D((2, 2)),                                                     # (32, 32, 8) -> (64, 64, 8)
            DeConv2D(16, (3, 3), activation='relu'),                                 # (64, 64, 8) -> (64, 64, 16)
            UnPooling2D((2, 2)),                                                     # (64, 64, 16) -> (128, 128, 16)
            DeConv2D(16, (3, 3), activation='relu', padding='same'),                 # (128, 128, 16) -> (128, 128, 16)
            UnPooling2D((2, 2)),                                                     # (128, 128, 16) -> (256, 256, 16)
            DeConv2D(3, (5, 5), activation='tanh', padding='same')                   # (256, 256, 16) -> (256, 256, 3)
        ])

        self.generator_autoencoder.compile(optimizer='adam', loss='mean_squared_error')

        self.generator_encoder = None
        self.generator_decoder = None

    def prepare_dataset(self, dataset_name):
        self.train, self.train_count = load_dataset_flow(dataset_name, "train", self.batch_size)
        self.test, self.test_count = load_dataset_flow(dataset_name, "test", self.batch_size)

        self.test_subset = map(
            lambda x: np.array(cv2.imread(path.join("dataset", dataset_name, "test", "images", x)))
                        .astype('float32') / 255,

            random.sample(os.listdir(path.join("dataset", dataset_name, "test", "images")), 5)
        )

    def fit(self):
        self.generator_autoencoder.fit_generator(
            self.train,
            steps_per_epoch=int(math.floor(self.train_count / self.batch_size)),
            epochs=15,
            validation_data=self.test,
            validation_steps=self.test_count,
            callbacks=[self.tensor_board]
        )
        self.separate_model()

    def train(self, data, weights=None):
        return self.generator_autoencoder.train_on_batch(data, data, weights)

    def load_model(self):
        self.generator_autoencoder = load_model('models/autoencoder_%s.h5' % self.name)
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

        self.generator_autoencoder.save('models/autoencoder_%s.h5' % self.name)

    def write_results(self):
        # Random Vector Image Creation
        for i in range(5):
            summary = tf.summary.image("Random Vector-%s %d" % (self.name, i),
                                       self.generator_decoder.predict(np.random.rand(1, 5, 5, 8)))

            self.tensor_board.writer.add_summary(summary.eval(session=backend.get_session()))

        # Auto Encoding Test
        for i in range(5):
            random_test = np.reshape(random.choice(self.test_subset), (1, 28, 28, 1))
            summary = tf.summary.image(
                "Auto-Encoding-%s %d" % (self.name, i),
                self.generator_autoencoder.predict(random_test),
                family="Auto-Encoding-%s %d" % (self.name, i)
            )
            self.tensor_board.writer.add_summary(summary.eval(session=backend.get_session()))
            self.tensor_board.writer.add_summary(tf.summary.image(
                "Auto-Encoding_Original-%s %d" % (self.name, i),
                random_test,
                family="Auto-Encoding-%s %d" % (self.name, i)
            ).eval(session=backend.get_session()))

        # Vector Walking
        for i in range(5):
            random_image = self.generator_encoder.predict(np.reshape(random.choice(self.test_subset), (1, 28, 28, 1)))
            transition_image = self.generator_encoder.predict(
                np.reshape(random.choice(self.test_subset), (1, 28, 28, 1)))
            transition_map = transition_image - random_image

            for j in range(10):
                for x, _ in enumerate(random_image[0]):
                    for y, __ in enumerate(random_image[0][x]):
                        random_image[0][x][y] += transition_map[0][x][y] / 10

                self.tensor_board.writer.add_summary(tf.summary.image(
                    'Vector-Space-Walking-{0}_{1}_{2}/10'.format(self.name, i, j),
                    self.generator_decoder.predict(random_image),
                    family="Vector-Space-Walking-%s_%d" % (self.name, i)
                ).eval(session=backend.get_session()))

        self.tensor_board.writer.close()


class Discriminator(object):
    # Deprecated

    def __init__(self):
        self.discriminator = Sequential([
            Conv2D(16, (3, 3),
                   activation='relu', padding='same', input_shape=(256, 256, 6)),  # (256, 256, 6) -> (256, 256, 16)
            MaxPooling2D((4, 4), padding='same'),                                  # (256, 256, 16) -> (64, 64, 16)
            Conv2D(8, (3, 3), activation='relu', padding='same'),                  # (64, 64, 16) -> (64, 64, 8)
            MaxPooling2D((4, 4), padding='same'),                                  # (64, 64, 8) -> (16, 16, 8)
            Conv2D((3, 3), activation='relu', padding='same'),                     # (16, 16, 8) -> (16, 16, 8)
            MaxPooling2D((4, 4), padding='same'),                                  # (16, 16, 8) -> (4, 4, 8)
            Dense(1),
            Activation('softmax')
        ])

        self.discriminator.compile(optimizer=Adam(rl=1e-3), loss='binary_crossentropy')

    def train(self, fake_data_b, real_data_b):
        legit = random.getrandbits(1)

        if bool(legit):
            x = tf.concat(fake_data_b, real_data_b)

        else:
            x = tf.concat(real_data_b, fake_data_b)

        return self.discriminator.train_on_batch(x, legit)

    def predict(self, fake_data_b, real_data_b):
        return self.discriminator.predict(tf.concat(fake_data_b, real_data_b))


# noinspection PyShadowingNames
class TransGANTrainer(object):
    def __init__(self, a_encoder, b_decoder, converter, discriminator, k_lambda=.001):
        self.a_encoder = a_encoder
        self.a_encoder.trainable = False
        self.a_dataset = None

        self.b_decoder = b_decoder
        self.b_decoder.trainable = False
        self.b_dataset = None

        self.converter = converter
        self.discriminator = discriminator

        self.generator = Sequential()
        self.generator.add(self.a_encoder)
        self.generator.add(self.converter)
        self.generator.add(self.b_decoder)

        self.generator_discriminator = Sequential()
        self.generator_discriminator.add(self.generator)
        self.generator_discriminator.add(self.discriminator)

        self.epsilon = backend.epsilon()
        self.k = self.epsilon
        self.k_lambda = k_lambda

    def prepare_dataset(self, a_generator, b_generator):
        self.a_dataset = a_generator
        self.b_dataset = b_generator

    # Referenced pbontrager/BEGAN-keras
    def train(self, epoch, data_size, batch_size, tensor_board, gamma=0.5):
        """
        :type epoch: int
        :type data_size: int
        :type batch_size: int
        :type tensor_board: TensorBoard
        :type gamma: float
        """
        batch_per_epoch = int(math.floor(data_size / batch_size))
        tensor_board.set_model(self.converter)

        for e in range(epoch):
            progress = Progbar(data_size)
            start = time.time()
            logs = {
                "M": 0,
                "Loss_D": 0,
                "Loss_G": 0,
                "k": 0
            }

            for batch in range(batch_per_epoch):
                # Unzip identical two data to one data

                real_a = self.a_dataset.next()[0]
                real_b = self.b_dataset.next()[0]

                fake_b = self.generator.predict(real_a)

                d_loss_real = self.discriminator.train(real_b)
                d_loss_gen = self.discriminator.train(fake_b, -self.k * np.ones(batch_size))
                d_loss = d_loss_real + d_loss_gen

                self.discriminator.trainable = False
                g_loss = self.generator_discriminator.train_on_batch(real_a, fake_b)
                self.discriminator.trainable = True

                self.k += self.k_lambda * (gamma * d_loss_real - g_loss)
                self.k = min(max(self.k, self.epsilon), 1)

                m_global = d_loss + np.abs(gamma * d_loss_real - g_loss)
                logs["M"] += m_global
                logs["Loss_D"] += d_loss
                logs["Loss_G"] += g_loss
                logs["k"] += self.k

                progress.add(batch_size, values=[
                    ("M", m_global),
                    ("Loss_D", d_loss),
                    ("Loss_G", g_loss),
                    ("k", self.k)
                ])

            logs = {k: v / batch_per_epoch for k, v in logs.items()}
            tensor_board.on_epoch_end(e, logs=logs)

            print('\nEpoch {}/{}, Time: {}'.format(e + 1, epoch, time.time() - start))

    def save_model(self):
        self.converter.save('models/converter.h5')
        self.discriminator.save('models/discriminator.h5')


# Move log files
print_section("Updating Logs")
check_directory_and_create("logs")
check_directory_and_create(path.join("logs", "autoencoder_photo"))
check_directory_and_create(path.join("logs", "autoencoder_monet"))
check_directory_and_create(path.join("logs", "old"))

for log_directory in os.listdir("logs"):
    current_log_directory = path.join("logs", log_directory)

    if not path.isdir(current_log_directory):
        continue

    if log_directory == 'old':
        continue

    for log_file in os.listdir(current_log_directory):
        current_log_file = path.join(current_log_directory, log_file)

        if not path.isfile(current_log_file):
            continue

        shutil.move(current_log_file, path.join("logs", "old", log_directory + " - " + log_file))

batch_size = 32
epoch = 50

# Training Auto Encoder
tensor_board = TensorBoardNoClosing(log_dir='logs/autoencoder_photo')
print_section("Auto Encoder Photo")
a_auto_encoder = AutoEncoder("photo", tensor_board, batch_size)
a_auto_encoder.prepare_dataset("photo")
a_auto_encoder.fit()
a_auto_encoder.separate_model()
a_auto_encoder.write_results()
tensor_board.finish()

tensor_board = TensorBoardNoClosing(log_dir='logs/autoencoder_monet')
print_section("Auto Encoder Monet")
b_auto_encoder = AutoEncoder("monet", tensor_board, batch_size)
b_auto_encoder.prepare_dataset("monet")
b_auto_encoder.fit()
b_auto_encoder.separate_model()
b_auto_encoder.write_results()
tensor_board.finish()

# Creating Converter and Discriminator
tensor_board = TensorBoardNoClosing(log_dir='logs/gan')
converter = Sequential([
    Conv2D(16, (4, 4), activation='relu', padding='same', input_shape=(32, 32, 8)),
    Conv2D(16, (3, 3), activation='relu', padding='same'),
    Conv2D(8, (3, 3), activation='relu', padding='same'),
    Conv2D(8, (2, 2), activation='relu', padding='same')
])
converter.compile(optimizer=Adam(rl=1e-4), loss='binary_crossentropy')()
discriminator = AutoEncoder("discriminator", None, batch_size)

# Training Converter and Discriminator
dataflow, data_size = load_dataset_flow("monet", "train", batch_size)

print_section("Converter, Discriminator")
trainer = TransGANTrainer(a_auto_encoder, b_auto_encoder, converter, discriminator)
trainer.prepare_dataset(load_dataset_flow("photo", "train", batch_size)[0], dataflow)
trainer.train(epoch, data_size, batch_size, tensor_board)
trainer.save_model()
tensor_board.finish()
