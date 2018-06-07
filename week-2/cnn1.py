import os
import numpy as np
from keras.models import Sequential
from keras.layers import Activation,Dropout,Flatten,Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Convolution2D,MaxPooling2D,ZeroPadding2D
from keras import optimizers

img_width = 150
img_height = 150

train_data_dir = 'C:/Users/yashc/Downloads/train'
test_data_dir = 'C:/Users/yashc/Downloads/validate'

datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(train_data_dir,target_size=(150, 150),batch_size=16,class_mode='binary')
test_generator = datagen.flow_from_directory(test_data_dir,target_size=(150, 150),batch_size=16,class_mode='binary')

#build CNN
model = Sequential()
model.add(Convolution2D(32,3,3,input_shape=(img_width,img_height,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32,3,3,input_shape=(img_width,img_height,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64,3,3,input_shape=(img_width,img_height,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])


nb_epoch = 30
nb_train_samples = 2048
nb_validation_samples = 832

model.fit_generator(train_generator,steps_per_epoch=nb_epoch,validation_data=test_generator,epochs=nb_epoch,validation_steps=nb_validation_samples)

model.save_weights('models/basic_cnn_30_epochs.h5')

model.evaluate_generator(test_generator,nb_validation_samples)
