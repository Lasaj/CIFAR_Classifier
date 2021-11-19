"""
CIFAR10 classifier
Author: Rick Wainwright
Date: 18/11/2021
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dropout, MaxPool2D, BatchNormalization, Flatten, Dense


target_names = ['aeroplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
img_height = 32
img_width = 32
img_channels = 3


def prepare_data():
    (train_x, train_y), (test_x, test_y) = cifar10.load_data()
    # normalise
    # train_x = tf.convert_to_tensor(train_x, dtype=float)
    # train_mean, train_sd = tf.nn.moments(train_x, axes=[1], keepdims=True)
    train_mean = np.mean(train_x)
    train_sd = np.std(train_x)
    train_x = train_x - train_mean
    train_x = train_x / train_sd

    # test_x = tf.convert_to_tensor(test_x, dtype=float)
    # test_mean, test_sd = tf.nn.moments(test_x, axes=[1], keepdims=True)
    test_mean = np.mean(test_x)
    test_sd = np.std(test_x)
    test_x = test_x - test_mean
    test_x = test_x / test_sd

    # one-hot encode labels
    train_y = tf.keras.utils.to_categorical(train_y, num_classes=10)
    test_y = tf.keras.utils.to_categorical(test_y, num_classes=10)
    return train_x, train_y, test_x, test_y


def augment_data(img, label):
    img = tf.image.resize_with_crop_or_pad(img, img_height + 8, img_width + 8)
    img = tf.image.random_crop(img, [img_height, img_width, img_channels])
    img = tf.image.random_flip_left_right(img)
    return img, label


def plot_performance(curves):
    # Plot the model performance
    # plot accuracy
    fig2, (gax1, gax2) = plt.subplots(1, 2)
    gax1.plot(curves.history['accuracy'])
    gax1.plot(curves.history['val_accuracy'])
    gax1.legend(['train', 'test'], loc='upper left')
    gax1.title.set_text("Accuracy")
    # plot loss
    gax2.plot(curves.history['loss'])
    gax2.plot(curves.history['val_loss'])
    gax2.legend(['train', 'test'], loc='upper left')
    gax2.title.set_text("Loss")
    fig2.savefig('acc_loss.png')


def get_model(filters=32):
    model = Sequential()
    model.add(Conv2D(filters, (3, 3), activation='relu', kernel_initializer='he_normal', input_shape=(32, 32, 3),
                     padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters*2, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters*2, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(filters*4, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters*4, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.4))

    model.add(Conv2D(filters*8, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters*8, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(filters*8, activation='relu', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Dense(filters*4, activation='relu', kernel_initializer='he_normal'))

    model.add(Dense(10, activation='softmax'))

    return model


model = get_model()
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9), loss='categorical_crossentropy',
              metrics=['accuracy'])

train_x, train_y, test_x, test_y = prepare_data()
train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y)).map(augment_data).shuffle(50000).batch(128)
test_ds = tf.data.Dataset.from_tensor_slices((test_x, test_y)).batch(128)

curves = model.fit(train_ds, epochs=60, validation_data=test_ds)
_, acc = model.evaluate(test_ds)

plot_performance(curves)

print('Accuracy > %.3f' % (acc * 100.0))
