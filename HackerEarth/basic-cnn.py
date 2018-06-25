import os
import cv2
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Activation,Dropout,Flatten,Dense,Convolution2D,MaxPooling2D,ZeroPadding2D
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.callbacks import EarlyStopping

train_data_dir = 'C:\\Users\\yashc\\Downloads\\hackerearth\\train'
test_data_dir = 'C:\\Users\\yashc\\Downloads\\hackerearth\\test'

def read_img(img_path):
    img = 

datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(train_data_dir,target_size=(150,150),batch_size=16,class_mode='categorical')
test_generator = datagen.flow_from_directory(test_data_dir,target_size=(150,150),batch_size=16,class_mode='categorical')
valid_generator = datagen.flow_from_directory(validate_data_dir,target_size=(150,150),batch_size=16,class_mode='categorical')

model = Sequential()
model.add(Convolution2D(32,3,3,input_shape=(150,150,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(64,3,3,input_shape=(150,150,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Convolution2D(128,3,3,input_shape=(150,150,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(29,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())

