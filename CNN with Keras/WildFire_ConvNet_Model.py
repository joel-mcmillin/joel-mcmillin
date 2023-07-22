#!/usr/bin/env python
# coding: utf-8

# Importing libraries
import tensorflow as tf
import numpy as np
from tensorflow import keras
import os
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt


# Rescaling training and test data and setting the training and test data
train = ImageDataGenerator(rescale=1/255)
test = ImageDataGenerator(rescale=1/255)

train_data = train.flow_from_directory('/forest_fire/Training and Validation/',
                                      target_size=(150, 150),
                                      batch_size=32,
                                      class_mode='binary')

test_data = test.flow_from_directory('/forest_fire/Testing/',
                                      target_size=(150, 150),
                                      batch_size=32,
                                      class_mode='binary')



# Confirming the class types are indexed correctly
test_data.class_indices



# Building the model using CNN
model = keras.Sequential()
model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(keras.layers.MaxPool2D(2, 2))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.MaxPool2D(2, 2))
model.add(keras.layers.Conv2D(128, (3, 3), activation='relu'))
model.add(keras.layers.MaxPool2D(2, 2))
model.add(keras.layers.Conv2D(128, (3, 3), activation='relu'))
model.add(keras.layers.MaxPool2D(2, 2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(512, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))



# Compiling the model and setting accuracy metrics
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])



# Visualizing the model with keras plot_model
from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)



# Visualizing the CNN layers, with legend, using visualkeras
import visualkeras
visualkeras.layered_view(model, legend=True) 
from PIL import ImageFont
font = ImageFont.truetype("arial.ttf", 20)
visualkeras.layered_view(model, legend=True, font=font)



# Model fit with 5 epochs
#r = model.fit(train_data, epochs=5, validation_data=test_data)
#96.24% accuracy w/ 5 epochs



# Model fit with 10 epochs
r = model.fit(train_data, epochs=10, validation_data=test_data)

# 98% accuracy w/o having added in image of cig smoke
# 99.13% accuracy w/ 10 epochs; 

'''
Below shows accuracy at 97.27-98.15% gives two of the highest value accuracy scores (97% and 95.6%)
Epoch 6/10
58/58 [==============================] - 20s 342ms/step - loss: 0.0886 - accuracy: 0.9727 - val_loss: 0.1042 - val_accuracy: 0.9706
Epoch 9/10
58/58 [==============================] - 22s 380ms/step - loss: 0.0607 - accuracy: 0.9815 - val_loss: 0.1741 - val_accuracy: 0.9559
'''


# Making predictions using our model
pred = model.predict(test_data)
pred = np.round(pred)

pred

print(len(pred))



# Plotting the loss and validation loss
# The Loss shows improvement across epochs with how the model fits training data
# The Validation Loss shows improvement across epochs until the last epoch, where
# the model does considerably worse in predicting using new data
import matplotlib.pyplot as plt

plt.plot(r.history['loss'], label='Loss')
plt.plot(r.history['val_loss'], label='Validation Loss')
plt.legend()



# Defining a function for predicting new images
def predictImage(filename):
    img1 = image.load_img(filename, target_size=(150, 150))
    plt.imshow(img1)
    y = image.img_to_array(img1)
    x = np.expand_dims(y, axis=0)
    val = model.predict(x)
    print(val)
    if val == 1:
        plt.xlabel("No Fire Detected", fontsize=25)
    elif val == 0:
        plt.xlabel("Fire Detected", fontsize=25)



# 1
predictImage('Testing/nofire/abc337.jpg')



# 2
predictImage('forest_fire/Download.jpg')



# 3
predictImage('forest_fire/dl2.jpg')



# 4
predictImage('forest_fire/dl3.jpg')



# 5
predictImage('forest_fire/dl4.jpg')



# 6
predictImage('forest_fire/dl5.jpg')



# 8
# The following image is perplexing, as it and image #4 come up as both Fire Detected and No Fire Detected
# across multiple instantiations of this model. This is an area of opportunity for modeling accuracy as 
# part of the goal of this project is early fire detection

predictImage('forest_fire/dl6.jpg')
