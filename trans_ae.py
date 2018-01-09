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

import cv2
import numpy as np
import tensorflow as tf

import math
import os
from os import path
import random
import shutil


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
        for i in range(7, 13):
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
class TransAETrainer(object):
    def __init__(self, b_ae, merge_ae, converter, batch_size):
        self.batch_size = batch_size

        self.a_dataset = None
        self.a_dataset_test = None

        self.data_size = 0
        self.data_size_test = 0

        self.b_decoder = b_ae.generator_decoder
        self.b_decoder.trainable = False

        self.merge_encoder = merge_ae.generator_encoder
        self.merge_encoder.trainable = False

        self.converter = converter

        self.generator = Sequential()
        self.generator.add(self.merge_encoder)
        self.generator.add(self.converter)
        self.generator.add(self.b_decoder)

        self.generator_encoder = Sequential()
        self.generator_encoder.add(self.generator)
        self.generator_encoder.add(self.merge_encoder)

    def prepare_dataset(self, a_name):
        self.a_dataset, self.data_size = load_dataset_flow(a_name, "train", self.batch_size)
        self.a_dataset_test, self.data_size_test = load_dataset_flow(a_name, "test", self.batch_size)

    def train(self, epoch, tensor_board):
        """
        :type epoch: int
        :type batch_size: int
        :type tensor_board: TensorBoard
        """

        def train_generator():
            while True:
                x = self.a_dataset.next()
                y = self.merge_encoder.predict(x)
                yield (x, y)

        def test_generator():
            while True:
                x = self.a_dataset_test.next()
                y = self.merge_encoder.predict(x)
                yield (x, y)

        batch_per_epoch = int(math.floor(self.data_size / self.batch_size))
        self.generator_encoder.fit_generator(
            train_generator(),
            steps_per_epoch=batch_per_epoch,
            epochs=epoch,
            validation_data=test_generator(),
            validation_steps=self.data_size_test,
            callbacks=[tensor_board]
        )

    def save_model(self):
        self.converter.save('models/converter.h5')

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
auto_encoders = {}

for name in ['monet', 'merged']:
    tensor_board = TensorBoardNoClosing(log_dir='logs/autoencoder_photo')
    print_section("Auto Encoder %s" % name[:1].upper() + name[1:])
    auto_encoder = AutoEncoder(name, tensor_board, batch_size)
    auto_encoder.prepare_dataset(name)
    auto_encoder.fit()
    auto_encoder.separate_model()
    auto_encoder.write_results()
    auto_encoders[name] = auto_encoder
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

# Training Converter
print_section("Converter")
trainer = TransAETrainer(auto_encoders['monet'], auto_encoders['merged'], converter, batch_size)
trainer.prepare_dataset("photo")
trainer.train(epoch, tensor_board)
trainer.save_model()
tensor_board.finish()
