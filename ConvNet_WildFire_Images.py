#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing necessary libraries
import tensorflow as tf
import numpy as np
from tensorflow import keras
import os
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt


# In[2]:


# Rescaling training and test data and setting the training and test data
train = ImageDataGenerator(rescale=1/255)
test = ImageDataGenerator(rescale=1/255)

train_data = train.flow_from_directory('C:/Users/joelm/OneDrive/Data Visualization/DSC680/DSC680_Project_2/forest_fire/Training and Validation/',
                                      target_size=(150, 150),
                                      batch_size=32,
                                      class_mode='binary')

test_data = test.flow_from_directory('C:/Users/joelm/OneDrive/Data Visualization/DSC680/DSC680_Project_2/forest_fire/Testing/',
                                      target_size=(150, 150),
                                      batch_size=32,
                                      class_mode='binary')


# In[3]:


# Confirming the class types are indexed correctly
test_data.class_indices


# In[4]:


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


# In[5]:


# Compiling the model and setting accuracy metrics
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[6]:


# Model fit with 5 epochs
#r = model.fit(train_data, epochs=5, validation_data=test_data)

#96.24% accuracy w/ 5 epochs


# In[6]:


# Model fit with 10 epochs
r = model.fit(train_data, epochs=10, validation_data=test_data)

# 98% accuracy w/o having added in image of cig smoke
# 99.13% accuracy w/ 10 epochs; 

'''
Below shows accuracy at 97.98-98.47% gives two of the highest value accuracy scores (97% and 95.6%)
Epoch 7/10
58/58 [==============================] - 21s 365ms/step - loss: 0.0679 - accuracy: 0.9798 - val_loss: 0.1194 - val_accuracy: 0.9706
Epoch 8/10
58/58 [==============================] - 22s 374ms/step - loss: 0.0471 - accuracy: 0.9847 - val_loss: 0.1805 - val_accuracy: 0.9559
'''


# In[7]:


# Making predictions using our model
pred = model.predict(test_data)
pred = np.round(pred)


# In[8]:


pred


# In[9]:


print(len(pred))


# In[22]:


# Plotting the loss and validation loss
# The Loss shows improvement across epochs with how the model fits training data
# The Validation Loss shows improvement across epochs until the last epoch, where
# the model does considerably worse in predicting using new data
import matplotlib.pyplot as plt

plt.plot(r.history['loss'], label='Loss')
plt.plot(r.history['val_loss'], label='Validation Loss')
plt.legend()


# In[11]:


# Defining a function for predicting new images
def predictImage(filename):
    img1 = image.load_img(filename, target_size=(150, 150))
    plt.imshow(img1)
    y = image.img_to_array(img1)
    x = np.expand_dims(y, axis=0)
    val = model.predict(x)
    print(val)
    if val == 1:
        plt.xlabel("No Fire Detected", fontsize=30)
    elif val == 0:
        plt.xlabel("Fire Detected", fontsize=30)


# In[12]:


predictImage('C:/Users/joelm/OneDrive/Data Visualization/DSC680/DSC680_Project_2/forest_fire/Testing/nofire/abc337.jpg')


# In[13]:


predictImage('C:/Users/joelm/OneDrive/Data Visualization/DSC680/DSC680_Project_2/forest_fire/Download.jpg')


# In[14]:


predictImage('C:/Users/joelm/OneDrive/Data Visualization/DSC680/DSC680_Project_2/forest_fire/dl2.jpg')


# In[20]:


predictImage('C:/Users/joelm/OneDrive/Data Visualization/DSC680/DSC680_Project_2/forest_fire/dl3.jpg')


# In[18]:


predictImage('C:/Users/joelm/OneDrive/Data Visualization/DSC680/DSC680_Project_2/forest_fire/dl4.jpg')


# In[19]:


predictImage('C:/Users/joelm/OneDrive/Data Visualization/DSC680/DSC680_Project_2/forest_fire/dl5.jpg')


# In[21]:


predictImage('C:/Users/joelm/OneDrive/Data Visualization/DSC680/DSC680_Project_2/forest_fire/dl6.jpg')


# In[ ]:




