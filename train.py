import tensorflow as tf
import numpy as np
import matplotlib as plt
import cv2
import os

mnist = tf.keras.datasets.fashion_mnist

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.95:
            print("\nReached 95% accuracy so cancelling training!")
            self.model.stop_training = True

callbacks = myCallback()

(images_train, labels_train), (images_test, labels_test) = mnist.load_data()

images_train, images_test = images_train / 255.0, images_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(units=512, activation="relu"),
    tf.keras.layers.Dense(units=512, activation="relu"),
    tf.keras.layers.Dense(units=256, activation="relu"),
    tf.keras.layers.Dense(units=256, activation="relu"),
    tf.keras.layers.Dense(units=10, activation="softmax")
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(images_train, labels_train, epochs=50, callbacks=[callbacks])

model.save(os.path.join("model.keras"))

model.evaluate(images_test, labels_test)
