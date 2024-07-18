#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 15:39:46 2024

@author: seanmilligan
"""
(dsfs) Seans-MacBook-Pro:~ seanmilligan$ cd /Users/seanmilligan/Desktop/Python/DSFS/
(dsfs) Seans-MacBook-Pro:DSFS seanmilligan$ tensorboard --logdir logs/fit


import tensorflow as tf
# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the pixel values to the range 0-1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Create a simple neural network model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Define a TensorBoard callback
log_dir = "/Users/seanmilligan/Desktop/Python/DSFS/logs/fit"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=3)

# Train the model with the TensorBoard callback
model.fit(x_train, y_train, epochs=1, validation_data=(x_test, y_test), callbacks=[tensorboard_callback])

#evaluate model
model.evaluate(x_test, y_test, verbose=2)








