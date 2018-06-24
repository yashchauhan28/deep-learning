import os
import numpy as np
from keras.models import Sequential
from keras.layers import Activation,Dropout,Flatten,Dense
from keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img
from keras.layers import Convolution2D,MaxPooling2D,ZeroPadding2D
from keras import optimizers
import tensorflow as tf
from scipy.misc import imresize
from keras.applications.imagenet_utils import preprocess_input, decode_predictions

img_width = 150
img_height = 150

train_data_dir = 'C:/Users/yashc/Downloads/train'
validate_data_dir = 'C:/Users/yashc/Downloads/validate'
test_data_dir = 'C:/Users/yashc/Downloads/test1'

datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(train_data_dir,target_size=(150, 150),batch_size=16,class_mode='categorical')
validate_generator = datagen.flow_from_directory(validate_data_dir,target_size=(150, 150),batch_size=16,class_mode='categorical')
test_generator = datagen.flow_from_directory(test_data_dir,target_size=(150,150),batch_size=16,class_mode='categorical')


def buildCNN(model):
    #build CNN
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
    model.add(Dense(2))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

def train(model):
    nb_epoch = 10
    nb_train_samples = 2048
    nb_validation_samples = 832
    model.fit_generator(train_generator,steps_per_epoch=nb_epoch,validation_data=validate_generator,epochs=nb_epoch,validation_steps=nb_validation_samples)
    #model.save_weights('models/basic_cnn_1_epochs.h5')

def loadModel(model):
    nb_validation_samples = 832
    model.load_weights('models/basic_cnn_30_epochs.h5')
    #print(model.metrics_names)
    #print(model.evaluate_generator(test_generator,nb_validation_samples))

def test(model):
    test_datagen = datagen
    img = load_img('C:/Users/yashc/Downloads/test1/18.jpg')
    img = imresize(img, size=(img_height, img_width))
    test_x = img_to_array(img).reshape(img_height, img_width,3)
    test_x = test_x.reshape((1,) + test_x.shape)
    test_generator = test_datagen.flow(test_x,
                                    batch_size=1,
                                       shuffle=False)
    prediction = model.predict_generator(test_generator, 1)[0][0]
    # test_img = tf.keras.preprocessing.image.load_img('C:/Users/yashc/Downloads/test1/5.jpg',target_size=(150,150))
    # x_test = np.array(test_img, np.float32) / 255
    # x_test = x_test[np.newaxis,...]
    # print(x_test)
    #x_test_norm = (test_img, np.float32) / 255
    #img = tf.keras.preprocessing.image.load_img('C:/Users/yashc/Downloads/test1/5.jpg',target_size=(150,150))
    #img = np.array(img)
    #img = img[np.newaxis,...]
    #img = np.expand_dims(np.array(img),axis = 0)
    #img = img[...,np.newaxis]
    #print(img)
    #prediction = model.predict(x_test)
    #predictions = np.argmax(prediction, axis= 1)
    print(prediction)
    #print('Predicted:', decode_predictions(prediction))

def main():
    model = Sequential()
    buildCNN(model)
    train(model)
    #loadModel(model)
    test(model)

if __name__ == '__main__':
    main()
