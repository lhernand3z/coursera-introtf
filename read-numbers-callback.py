#Week 2: Implementing Callbacks in TensorFlow using the MNIST Dataset
#
# In the course you learned how to do classification using Fashion MNIST, a data set containing items of clothing.
# There's another, similar dataset called MNIST which has items of handwriting -- the digits 0 through 9.
#
# Write an MNIST classifier that trains to 99% accuracy and stops once this threshold is achieved.
# In the lecture you saw how this was done for the loss but here you will be using accuracy instead.
#
# Some notes:
#
#     Your network should succeed in less than 9 epochs.
#     When it reaches 99% or greater it should print out the string "Reached 99% accuracy so cancelling training!"
#     and stop training.
#     If you add any additional variables, make sure you use the same names as the ones used in the class. This is
#     important for the function signatures (the parameters and names) of the callbacks.
#

import os
import tensorflow as tf
from tensorflow import keras

#Load the data
# Get current working directory
current_dir = os.getcwd()

# Append data/mnist.npz to the previous path to get the full path (need to figure out how to download it)
# The file mnist.npz is already included in the current workspace under the data directory.
# By default the load_data from Keras accepts a path relative to ~/.keras/datasets but in this case it is stored somewhere else,
# as a result of this, you need to specify the full path.

data_path = os.path.join(current_dir, "data/mnist.npz")

# Discard test set, not needed for this exercise
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data(path=data_path)

# Normalize pixel values
x_train = x_train / 255.0

data_shape = x_train.shape
print(f"There are {data_shape[0]} examples with shape ({data_shape[1]}, {data_shape[2]}")
#Sample output
# There are 60000 examples with shape (28, 28)

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') is not None and logs.get('accuracy') > 0.99:
            print("\nReached 99% accuracy so cancelling training!")

            #Stop training
            self.model.stop_training = True

def train_mnist(x_train, y_train):

    # Instantiate the callback class
    callbacks = myCallback()

    # Define the model
    model = tf.keras.models.Sequential([ tf.keras.layers.Flatten(),
                                         tf.keras.layers.Dense(512, activation=tf.nn.relu),
                                         tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Fit the model for 10 epochs and save the training history
    history = model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])

    return history


