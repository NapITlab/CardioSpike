#!/usr/bin/env python
# coding: utf-8


import time
from keras import backend as K
import os
from PIL import Image
from tqdm import tqdm, tqdm_notebook
import tensorflow as tf

import numpy as np
import os
from PIL import Image
import pandas as pd
import json
from os import listdir, path
import re
import cv2
import keras
from keras import backend as K
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, UpSampling2D
from keras.optimizers import Adam
from keras.models import Model, load_model
from keras.utils import CustomObjectScope
from keras.layers import Conv2D, UpSampling2D, Input, MaxPooling2D, Dropout, concatenate, Conv2DTranspose
from keras.layers import Conv1D, UpSampling1D, Input, MaxPooling1D, Dropout, concatenate
from keras.losses import binary_crossentropy
import matplotlib.pyplot as plt


data_all = []
data_all_y = []
for id_i in pd.unique(data['id']):
    a = data[data['id']==id_i]
    a = a.sort_values(by=['time'])
    a.reset_index(inplace=True)
    print(len(a))
    for i in range(32, len(a) - 32, 16):
        x=a.iloc[i-32:i+32]['x']
        x = (x - x.mean())/x.std()
        y = a.iloc[i-32:i+32]['y']
        X = pd.DataFrame(list(x)).T
        Y = pd.DataFrame(list(y)).T
        X['id'] = id_i
        if len(data_all)==0:
            data_all = X
            data_all_y = Y
        else:
            data_all = data_all.append(X)
            data_all_y = data_all_y.append(Y)


# In[ ]:


data_all.columns = [str(i) + '_x' for i in data_all.columns]
data_all_y.columns = [str(i) + '_y' for i in data_all_y.columns]


# In[ ]:


data_all = pd.concat([data_all, data_all_y], axis=1)


# In[ ]:


a = pd.DataFrame(pd.unique(data['id'])).sample(frac=1, random_state=1)

a_train = a.iloc[:int(229*0.8)][0]
a_test = a.iloc[int(229*0.8):][0]


# In[ ]:


a_train = pd.DataFrame(a_train)
a_test = pd.DataFrame(a_test)

a_train.columns = ['id_x']
a_test.columns = ['id_x']


# In[ ]:


data_all_train = pd.merge(data_all, a_train, left_on=['id_x'], right_on=['id_x'])
data_all_test = pd.merge(data_all, a_test, left_on=['id_x'], right_on=['id_x'])


# In[ ]:


Y_train = data_all_train[data_all.columns[np.array(['_y' in i for i in data_all.columns])]]
Y_test = data_all_test[data_all.columns[np.array(['_y' in i for i in data_all.columns])]]


# In[ ]:


X_train = data_all_train[data_all.columns[np.array([('_x' in i) & ('id_x' not in i) for i in data_all.columns])]]
X_test = data_all_test[data_all.columns[np.array([('_x' in i) & ('id_x' not in i) for i in data_all.columns])]]


# In[ ]:


batch_size = 64


# In[ ]:


def generator_train(data_all_train=data_all_train, i=0, batch_size = batch_size):
    while True:
        image1 = np.zeros((batch_size, 64, 1))
        target = np.zeros((batch_size, 64, 1))
        for k in range(batch_size):
            image1[k, : :] = np.array(X_train.iloc[i]).reshape(-1, 1)
            target[k] = np.array(Y_train.iloc[i]).reshape(-1, 1)

            i += 1
        if i>=len(X_train)-batch_size+1:
            i = 0
        yield image1, target


# In[ ]:


def generator_test(data_all_test = data_all_test, i=0, batch_size = batch_size):
    while True:
        image1 = np.zeros((batch_size, 64, 1))
        target = np.zeros((batch_size, 64, 1))
        for k in range(batch_size):
            image1[k, : :] = np.array(X_test.iloc[i]).reshape(-1, 1)
            target[k] = np.array(Y_test.iloc[i]).reshape(-1, 1)

            i += 1
        if i>=len(X_test)-batch_size+1:
            i = 0
#             data_all_train = data_all_train.sample(frac=1)
        yield image1, target


# In[ ]:


keras.backend.clear_session()

import keras.backend as K
from keras.layers import Conv2DTranspose, Lambda


def Conv1DTranspose(input_tensor, filters, kernel_size, strides=2, padding='same'):
    """
        input_tensor: tensor, with the shape (batch_size, time_steps, dims)
        filters: int, output dimension, i.e. the output tensor will have the shape of (batch_size, time_steps, filters)
        kernel_size: int, size of the convolution kernel
        strides: int, convolution step size
        padding: 'same' | 'valid'
    """
    x = Lambda(lambda x: K.expand_dims(x, axis=2))(input_tensor)
    x = Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1), strides=(strides, 1), padding=padding)(x)
    x = Lambda(lambda x: K.squeeze(x, axis=2))(x)
    return x



# create model

inputs = Input((64, 1))
conv1 = Conv1D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
conv1 = Conv1D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
drop1 = Dropout(0.45)(conv1)

pool1 = Conv1D(16, 3, strides=2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop1)
conv2 = Conv1D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
conv2 = Conv1D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
drop2 = Dropout(0.45)(conv2)

pool2 = Conv1D(64, 3, strides=2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop2)
conv3 = Conv1D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
conv3 = Conv1D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
drop3 = Dropout(0.45)(conv3)

pool3 = Conv1D(64, 3, strides=2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop3)
conv4 = Conv1D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
conv4 = Conv1D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
drop4 = Dropout(0.45)(conv4)

pool4 = Conv1D(128, 3, strides=2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop4)

conv5 = Conv1D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
conv5 = Conv1D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
drop5 = Dropout(0.45)(conv5)

up6 = Conv1DTranspose(drop5, 128, 2, strides=2, padding = 'same')
merge6 = concatenate([drop4,up6])
conv6 = Conv1D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
conv6 = Conv1D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
drop6 = Dropout(0.45)(conv6)

up7 = Conv1DTranspose(drop6, 64, 2, strides=2, padding = 'same')
merge7 = concatenate([conv3,up7])
conv7 = Conv1D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
conv7 = Conv1D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

up8 = Conv1DTranspose(conv7, 32, 2, strides=2, padding = 'same')
merge8 = concatenate([conv2,up8])
conv8 = Conv1D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
conv8 = Conv1D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

up9 = Conv1DTranspose(conv8, 16, 2, strides=2, padding = 'same')
merge9 = concatenate([conv1,up9])
conv9 = Conv1D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
conv9 = Conv1D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

conv10 = Conv1D(1, 1, activation = 'sigmoid')(conv9)

model = Model(inputs = inputs, outputs = conv10)

model.summary()


model.compile(loss='binary_crossentropy', metrics=['acc'], optimizer=optimizers.RMSprop(lr=1e-5))

train_size = X_train.shape[0]//batch_size
test_size = X_test.shape[0]//batch_size

callbacks_list=[
                keras.callbacks.EarlyStopping(monitor='val_loss', patience=1500),
                keras.callbacks.ModelCheckpoint(filepath='best3.h5', monitor='val_loss', 
                                                save_best_only=True, mode='min')
                ]

history = model.fit_generator(
            generator_train(),
            steps_per_epoch=train_size,
            epochs=10000,
            validation_data=generator_test(),
            validation_steps=test_size,
            callbacks=callbacks_list
        )
