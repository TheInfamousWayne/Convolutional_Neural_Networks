# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 02:17:48 2017

@author: Vaibhav
"""

# Importing the libraries
from keras.models import Sequential # for initialising the CNN as a sequence of layers
from keras.layers import Conv2D # 2D and not 3D because we dont have video. just images
from keras.layers import MaxPooling2D
from keras.layers import Flatten 
from keras.layers import Dense # to make fully connected ANN

# Creating the CNN
classifier = Sequential()

# Performing convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu')) # adding the input_shape because it is unknown initially

# Performing Max Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second Convolution and MaxPooling layer
classifier.add(Conv2D(32, 3, 3, activation = 'relu')) #no need to add input_shape option since it is already known to the network (as it is being continued from before)
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Flattening
classifier.add(Flatten())

# Creating ANN (Full Connection)
classifier.add(Dense(units = 128, activation = 'relu')) #hidden layer. Units is for no. of hidden layer (from inpt layer to hidden layer) nodes
classifier.add(Dense(units = 1, activation = 'sigmoid')) #output layer. output is 1 node since we have a binary classifier.

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

###############################################################################

# Data Augmentation
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255) # just image rescaling for test_set. No augmentation in test_set needed.

# Training Set
training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

# Testing Set
test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

# Fitting the CNN on training set and checking accuracy with test set
classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test_set,
        validation_steps=2000)
# here we give training set, the number of elements in each epoch = 8000 = contents of training set; 
# then we give test set and the no. of elements in the test set

















