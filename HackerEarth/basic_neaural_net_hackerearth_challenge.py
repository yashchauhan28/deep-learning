# coding: utf-8
import os
import cv2
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Activation,Dropout,Flatten,Dense,Convolution2D,MaxPooling2D,ZeroPadding2D
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.callbacks import EarlyStopping
base_dir = 'C:\\Users\\yashc\\Downloads\\hackerearth'
train_file = 'C:\\Users\\yashc\\Downloads\\hackerearth\\meta-data\\train.csv'
test_file = 'C:\\Users\\yashc\\Downloads\\hackerearth\\meta-data\\test.csv'

train_data_dir = 'C:\\Users\\yashc\\Downloads\\hackerearth\\train'
test_data_dir = 'C:\\Users\\yashc\\Downloads\\hackerearth\\test'

img_width = 256
img_height = 256
train = pd.read_csv(train_file)
test = pd.read_csv(test_file)

train_data = []
test_data = []
train_labels = train['Animal'].values
with open(base_dir + '\\' + 'train_data_pickleFile.pickle', 'rb') as handle:
    train_data = pickle.load(handle)
with open(base_dir + '\\' + 'test_data_pickleFile.pickle', 'rb') as handle:
    test_data = pickle.load(handle)
    
x_train = np.array(train_data,np.float32) / 255.
x_test = np.array(test_data,np.float32) / 255.
label_list = train['Animal'].tolist()
Y_train = {k:v+1 for v,k in enumerate(set(label_list))}
y_train = [Y_train[k] for k in label_list]
y_train = to_categorical(y_train)
model = Sequential()
model.add(Convolution2D(32,(3,3),input_shape=(img_width,img_height,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(64,3,3,input_shape=(img_width,img_height,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Convolution2D(128,3,3,input_shape=(img_width,img_height,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(y_train.shape[1],activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())
early_stops = EarlyStopping(patience=3,monitor='val_acc')
model.fit(x_train,y_train,batch_size=10,epochs=1,validation_split=0.3,callbacks=[early_stops])
model.fit(x_train,y_train,batch_size=50,epochs=15,validation_split=0.3,callbacks=[early_stops])
model.fit(x_train,y_train,batch_size=20,epochs=15,validation_split=0.3,callbacks=[early_stops])
model.fit(x_train,y_train,batch_size=25,epochs=15,validation_split=0.3)
model.save_weights(base_dir + '\\' + 'epoch-15.h5')
predictions = model.predict(x_test)
predictions
predictions2 = np.argmax(predictions,axis = 1)
predictions2
y_maps = dict()
y_maps = {v:k for k,v in Y_train.items()}
pred_labels = [y_maps[k] for k in predictions]
y_maps = dict()
y_maps = {v:k for k,v in Y_train.items()}
pred_labels = [y_maps[k] for k in predictions]
y_maps
pred_labels = [y_maps[k] for k in predictions]
ymaps[1]
y_maps[1]
pred_labels = []
for k in predictions:
    pred_labels.append(y_maps[k])
    
for i in range(1,31):
    pred_labels.append(y_maps[i])
    
pred_labels
pred_labels = [y_maps[k] for k in predictions2]
for i in range(5):
    print('I see this product is {}'.format(pred_labels[i]))
    plt.imshow(read_image(TEST_PATH +'{}.png'.format(test.image_id[i])))
    plt.show()
    
import matplotlib.pyplot as plt
for i in range(5):
    print('I see this product is {}'.format(pred_labels[i]))
    
predictions
predictions[1]
predictions[2]
predictionsxx = predictions
np.round(predictions,5)
np.round(predictions,4)
np.set_printoptions(supress = True)
np.set_printoptions(suppress = True)
predictions
answer = {}
answer['image_id'] = test.image_id
get_ipython().run_line_magic('save', 'basic_neaural_net_hackerearth_challenge.py')
get_ipython().run_line_magic('save', 'basic_neaural_net_hackerearth_challenge')
get_ipython().run_line_magic('save', '-f basic_neaural_net_hackerearth_challenge.py')
import readline
get_ipython().run_line_magic('save', 'basic_neaural_net_hackerearth_challenge.py')
get_ipython().run_line_magic('save', 'basic_neaural_net_hackerearth_challenge.py 1-50')
