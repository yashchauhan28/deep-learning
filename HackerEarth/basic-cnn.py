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

def read_img(img_path):
    img = cv2.imread(img_path,cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_width,img_height))
    return img


train = pd.read_csv(train_file)
test = pd.read_csv(test_file)

train_data = []
test_data = []
train_labels = train['Animal'].values

# for img in tqdm(train['Image_id'].values):
#     train_data.append(read_img(train_data_dir + '\\' + img))

# for img in tqdm(test['Image_id'].values):
#     test_data.append(read_img(test_data_dir + '\\' + img))

# with open('train_data_pickleFile.pickle', 'wb') as handle:
#     pickle.dump(train_data,handle,protocol=pickle.HIGHEST_PROTOCOL)
# with open('test_data_pickleFile.pickle', 'wb') as handle:
#     pickle.dump(test_data,handle,protocol=pickle.HIGHEST_PROTOCOL)

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
model.save_weights(base_dir + '\\' + 'epoch-15.h5')
predictions = model.predict(x_test)
predictions = np.argmax(predictions,axis = 1)

y_maps = dict()
y_maps = {v:k for k,v in Y_train.items()}
pred_labels = [y_maps[k] for k in predictions]

sub1 = pd.DataFrame({'Image_Id': test.image_id, 'Label': pred_labels})
sub1.to_csv('sub_one.csv',index=False)