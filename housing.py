#Assignment from Coursera, week 1, 'Introduction to Tensorflow for AI
import tensorflow as tf
import numpy as np
from tensorflow import keras

def house_model():

    # Define input tensors manually for houses with 1 up to 6 bedrooms
    # Hint: Remember to explicitly set the dtype as float
    xs = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    ys = np.array([50, 100, 150, 200, 250, 300, 350])

    # Scaling down ys, for some reason they say it's better for the modeling
    # maybe number crunching is easier ?
    ys = ys / 100
    print('DEBUG: Just want to see the array', ys)

    # Define the model (should be a model with 1 dense layer and 1 unit)
    model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

    # Compile your model
    # Set the optimizer to Stochastic Gradient Descent
    # and use Mean Squared Error as the loss function
    model.compile(optimizer='sgd', loss='mean_squared_error')

    # Train your model for 1000 epoch by feeding the i/o tensors
    model.fit(xs, ys, epochs=1000)

    return model

#Get your trained model
model = house_model()

#Get your prediction
new_y = 7.0
prediction = model.predict([new_y])[0]
print(prediction)



