#Copy of Week2 colab from Coursera Intro to Tensorflow
#this is an example on computer vision but with a using a callback class to stop
#the training once a certain level of accuracy

import tensorflow as tf
print(tf.__version__)


#Label	Description
# 0	T-shirt/top
# 1	Trouser
# 2	Pullover
# 3	Dress
# 4	Coat
# 5	Sandal
# 6	Shirt
# 7	Sneaker
# 8	Bag
# 9	Ankle boot


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        '''
        Halts the training after reaching 60 percent accuracy

        Args:
          epoch (integer) - index of epoch (required but unused in the function definition below)
          logs (dict) - metric results from the training epoch
        '''
        if(logs.get('accuracy') >= 0.88):
            print("\nReached 88% accuracy so cancelling training!")
            self.model.stop_training = True

callbacks = myCallback()

#Load the Fashion MNIST dataset
fmnist = tf.keras.datasets.fashion_mnist

#Load the training and test split of the Fashion MNIST dataset
(training_images, training_labels), (test_images, test_labels) = fmnist.load_data()

#Normalize the pixel values of the train and test images
training_images = training_images / 255.0
test_images = test_images / 255.0

# Build the classification model
# Flatten: takes the square(array) of the picture and turns it into a 1-dimensional array.
#  Rule of Thumb: the first layer should be the same shape as your data. 28x28 would need a 28layers of 28 neurons
#     unfeasible, so we flatten it to 28x28=784x1
# Dense: Adds a layer of neurons,
#  Rule of Thumb: the last layer should match the number of classes you're classifying
# activation, each layer of neurons needs an activation function to tell them what to do:
#   ReLU, it only passes values 0 or greater to the next layer
#   Softmax, takes a list of values and scales these so the sum of all elements will be equal to 1
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(512, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

# Compile the module
model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

# Train the module
model.fit(training_images, training_labels, epochs=5, callbacks=[callbacks])

# Evaluate the model with unseen data
model.evaluate(test_images, test_labels)

# Predicts the classifications of the test_images
classifications = model.predict(test_images)
print(classifications[0])

print(classifications[10])

print(classifications[42])

