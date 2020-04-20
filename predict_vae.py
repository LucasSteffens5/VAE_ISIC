# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 11:24:18 2020

@author: lucas
"""

from keras.models import load_model

import keras
from keras.layers import Conv2D, Conv2DTranspose, Input, Flatten, Dense, Lambda, Reshape
from keras.layers import BatchNormalization
from keras.models import Model
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from keras.preprocessing import image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import cv2
from keras.preprocessing.image import array_to_img
from keras.preprocessing.image import save_img


img_width, img_height =256,256



def files_path04(path):  # Percorre todo diretorio e pega o nome dos arquivos
    filenames=[]
    for p, _, files in os.walk(os.path.abspath(path)):
        for file in files:
            filenames.append( os.path.join(p, file))
    return filenames


def extract_vector(path):
    i=0
    path1= files_path04(path +'benign\\')
    path2= files_path04(path +'malignant\\')
    dim = len(path1) + len(path2)
    imags = np.ndarray((dim,img_width, img_height,3), dtype=np.uint8)
    targets = np.ndarray((dim,), dtype=np.uint8)
    for im in path1:
        #print(im)
        
        img = image.load_img(im, target_size=(img_width, img_height))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        imags[i]=x
        targets[i] = 0
        i+=1
    for im in path2:
        #print(im)
        
        img = image.load_img(im, target_size=(img_width, img_height))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        imags[i]=x
        targets[i] = 1
        i+=1
        
        
    return imags, targets

path = 'C:\\Users\\lucas\\Desktop\\Codes_TCC\\Testes_Inception\\BaseBM\\testerapido\\'





batch_size = 32
no_epochs = 1000
validation_split = 0.2
verbosity = 1
latent_dim = 800
num_channels = 3


input_train,target_train = extract_vector(path)
input_test,target_test =  extract_vector(path)
input_shape = (img_height, img_width, num_channels)


input_train = input_train.astype('float32')
input_test = input_test.astype('float32')





input_train = input_train / 255
input_test = input_test / 255




i       = Input(shape=input_shape, name='encoder_input')
cx      = Conv2D(filters=8, kernel_size=3, strides=2, padding='same', activation='relu')(i)
cx      = BatchNormalization()(cx)
cx      = Conv2D(filters=16, kernel_size=3, strides=2, padding='same', activation='relu')(cx)
cx      = BatchNormalization()(cx)
x       = Flatten()(cx)
x       = Dense(20, activation='relu')(x)
x       = BatchNormalization()(x)
mu      = Dense(latent_dim, name='latent_mu')(x)
sigma   = Dense(latent_dim, name='latent_sigma')(x)








def kl_reconstruction_loss(true, pred):
  
  reconstruction_loss = keras.losses.mean_squared_error(K.flatten(true), K.flatten(pred)) * img_width * img_height
  print('Loss reconstrution: \n')
  print(reconstruction_loss)
  
  kl_loss = 1 + sigma - K.square(mu) - K.exp(sigma)
  kl_loss = K.sum(kl_loss, axis=-1)
  kl_loss *= -0.5
  print('Loss KL: \n')
  print(kl_loss)
  
  print('Loss somada: \n')
  print(K.mean(reconstruction_loss + kl_loss))
  return K.mean(reconstruction_loss + kl_loss)


vaee = load_model("vae.h5",custom_objects={'kl_reconstruction_loss': kl_reconstruction_loss})
print('Load \n\n\n\n\n\n')
vaee.summary()

output = vaee.predict(input_test)

a= array_to_img(output3])
a.show()
