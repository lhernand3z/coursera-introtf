#Copy of Week2 colab from Coursera Intro to Tensorflow
#this is an example on computer vision

import tensorflow as tf
print(tf.__version__)

#Load the Fashion MNIST dataset
fmnist = tf.keras.datasets.fashion_mnist

#Load the training and test split of the Fashion MNIST dataset
(training_images, training_labels), (test_images, test_labels) = fmnist.load_data()

#Printing a training image/label, just to see the data

import numpy as np
import matplotlib.pyplot as plt

#You can put between 0 and 59999 here
index = 42

# Set number of characters per row when printing
np.set_printoptions(linewidth=320)

# Print the label and image
print(f'LABEL: {training_labels[index]}')
print(f'\nIMAGE PIXEL ARRAY:\n {training_images[index]}')

#Visualize the image
plt.imshow(training_images[index])

#Normalize the pixel values of the train and test images
training_images = training_images / 255.0
test_images = test_images / 255.0

# Build the classification model
# Flatten: takes the square(array) of the picture and turns it into a 1-dimensional array
# Dense: Adds a layer of neurons
# activation, each layer of neurons needs an activation function to tell them what to do:
#   ReLU, it only passes values 0 or greater to the next layer
#   Softmax, takes a list of values and scales these so the sum of all elements will be equal to 1
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

# Compile the module
model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

# Train the module
model.fit(training_images, training_labels, epochs=5)

# Evaluate the model with unseen data
model.evaluate(test_images, test_labels)

