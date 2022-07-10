import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
import random
import cv2 as cv
import numpy as np
from PIL import Image

# import pathlib
# dataset_url = 'Tumbs-Up'
# data_dir = tf.keras.utils.get_file(origin=dataset_url,
#                                    fname='thumbsUp',
#                                    untar=True)
# data_dir = pathlib.Path(data_dir)
# image_count = len(list(data_dir.glob('*/*.jpg')))
# print(image_count)

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)
target_size = (224, 224)

train_generator = train_datagen.flow_from_directory(
    'Thumbs-Up',
    target_size=target_size,
    batch_size=32)


dropout_rate = 0.5
batch_size = 128
epochs = 2
model_version = 2

model2 = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)

model2.summary()
# compile model
model2.compile(optimizer="adam",
               loss="categorical_crossentropy",
               metrics=["accuracy"])

model2.fit(train_generator, epochs=epochs)

# model evaluation
print("\nTraining Set: ")
model2.evaluate(train_generator, verbose=2)
#print("\nTest Set: ")
#model2.evaluate(xtest,  ytest, verbose=2)

