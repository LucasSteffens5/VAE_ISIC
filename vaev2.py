# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 17:35:45 2020

@author: lucas

  Variational Autoencoder (VAE).
  VAE desitando a base de dados ISIC, onde as imagens foram separadas em imagens benignas com presença de curativos nas imagens e malignas sem curativos na imagem.
  
  
  Baseado em: https://keras.io/examples/variational_autoencoder_deconv/
  e em : Https://www.machinecurve.com/index.php/2019/12/30/how-to-create-a-variational-autoencoder-with-keras/
"""



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
from keras import optimizers

img_width, img_height =256,256 # Imagens maiores causam estouro da memoria da placa de video



def files_path04(path):  # Percorre todo diretorio e pega o nome dos arquivos
    filenames=[]
    for p, _, files in os.walk(os.path.abspath(path)):
        for file in files:
            filenames.append( os.path.join(p, file))
    return filenames


def extract_vector(path):# Extrai os vetores  das imagens 
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
        targets[i] = 0  #Beniga e com curativos
        i+=1
    for im in path2:
        #print(im)
        
        img = image.load_img(im, target_size=(img_width, img_height))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        imags[i]=x
        targets[i] = 1  # Maliga e sem curativos
        i+=1
        
        
    return imags, targets
# Caminho das pastas de treino e teste
path = 'C:\\Users\\lucas\\Desktop\\Codes_TCC\\Testes_Inception\\BaseBM\\train\\'
path1 = 'C:\\Users\\lucas\\Desktop\\Codes_TCC\\Testes_Inception\\BaseBM\\validation\\'



# Configuraçao modelos e dados

batch_size = 32
no_epochs = 1000
validation_split = 0.2
verbosity = 1
latent_dim = 1000
num_channels = 3

# Reshape data
input_train,target_train = extract_vector(path)
input_test,target_test =  extract_vector(path1)
input_shape = (img_height, img_width, num_channels)

## Parse numbers as floats
input_train = input_train.astype('float32')
input_test = input_test.astype('float32')




# Normalize data
input_train = input_train / 255
input_test = input_test / 255





#Encoder
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

# Salva a dimensão da conV2d pra usar no decodificador
conv_shape = K.int_shape(cx)

# Define sampling with reparameterization trick
def sample_z(args):
  mu, sigma = args
  batch     = K.shape(mu)[0]
  dim       = K.int_shape(mu)[1]
  eps       = K.random_normal(shape=(batch, dim))#, mean=0., stddev=0.1)
  return mu + K.exp(sigma / 2) * eps


z       = Lambda(sample_z, output_shape=(latent_dim, ), name='z')([mu, sigma])


encoder = Model(i, [mu, sigma, z], name='encoder')
encoder.summary()

#Decoder


d_i   = Input(shape=(latent_dim, ), name='decoder_input')
x     = Dense(conv_shape[1] * conv_shape[2] * conv_shape[3], activation='relu')(d_i)
x     = BatchNormalization()(x)
x     = Reshape((conv_shape[1], conv_shape[2], conv_shape[3]))(x)

cx    = Conv2DTranspose(filters=16, kernel_size=3, strides=2, padding='same', activation='relu')(x)
cx    = BatchNormalization()(cx)
cx    = Conv2DTranspose(filters=8, kernel_size=3, strides=2, padding='same',  activation='relu')(cx)
cx    = BatchNormalization()(cx)

o     = Conv2DTranspose(filters=num_channels, kernel_size=3, activation='sigmoid', padding='same', name='decoder_output')(cx)


decoder = Model(d_i, o, name='decoder')
decoder.summary()



# VAE
vae_outputs = decoder(encoder(i)[2])
vae         = Model(i, vae_outputs, name='vae')
vae.summary()

# Loss
def kl_reconstruction_loss(true, pred):
  # Reconstrução loss
  reconstruction_loss = keras.losses.mean_squared_error(K.flatten(true), K.flatten(pred)) * img_width * img_height
  print('Loss reconstrution: \n')
  print(reconstruction_loss)
  # KL divergence loss
  kl_loss = 1 + sigma - K.square(mu) - K.exp(sigma)
  kl_loss = K.sum(kl_loss, axis=-1)
  kl_loss *= -0.5
  print('Loss KL: \n')
  print(kl_loss)
  # Total loss = 50% recontrução + 50% KL divergence loss
  print('Loss somada: \n')
  print(K.mean(reconstruction_loss + kl_loss))
  return K.mean(reconstruction_loss + kl_loss)

#  VAE
vae.compile(optimizer=Adam(lr=1e-5), loss=kl_reconstruction_loss)

mcp= ModelCheckpoint(filepath = 'vaemodelcheckpoint.h5',monitor='loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=5)

rlr = ReduceLROnPlateau(monitor = 'loss', factor = 0.2, epochs = 10, verbose = 1)
es = EarlyStopping(monitor =  'loss', min_delta = 1e-10, patience = 15, verbose = 1)


# Treino
history = vae.fit(input_train, input_train, epochs = no_epochs, batch_size = batch_size, validation_split = validation_split,callbacks = [es, rlr, mcp])
vae.save_weights('vaepesosisic.h5')
vae.save('vae.h5')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


#Visualização
def viz_latent_space(encoder, data):
  input_data, target_data = data
  mu, _, _ = encoder.predict(input_data)
  print(mu)
  print(mu.shape)
  plt.figure(figsize=(12, 10))
  plt.scatter(mu[:, 0], mu[:, 1], c=target_data)
  plt.xlabel('z - dim 1')
  plt.ylabel('z - dim 2')
  plt.colorbar()
  plt.show()
  #plt.imshow(mu[0][:,:,0])


def visualize(encoder,decoder,data): # Devolve os vetores decodificados e codificados
     
     x_test, y_test = data
     
     z_mean, a, b = encoder.predict(x_test)
     
     
     x_decoded = decoder.predict(z_mean)
     
     
     return z_mean,x_decoded 

def printa(b):  # Salva as imagens decodificadas
    i= 0
    
    while i<len(b):
        
        a= array_to_img(b[i])
        save_img(str(latent_dim)+'reconstuido_'+str(i)+'.jpg', a)
        
        i+=1
         

def viz_decoded(encoder, decoder, data): # Gera amostras saidas do decodificador
  num_samples = 15
  figure = np.zeros((img_width * num_samples, img_height * num_samples, num_channels))
  grid_x = np.linspace(-4, 4, num_samples)
  grid_y = np.linspace(-4, 4, num_samples)[::-1]
  for i, yi in enumerate(grid_y):
      for j, xi in enumerate(grid_x):
          
          z_sample =np.zeros((1,latent_dim))#np.array([[xi, yi]])#*0.1
          
          x_decoded = decoder.predict(z_sample)
          digit = x_decoded[0].reshape(img_width, img_height, num_channels)
          figure[i * img_width: (i + 1) * img_width,
                  j * img_height: (j + 1) * img_height] = digit
  plt.figure(figsize=(10, 10))
  start_range = img_width // 2
  end_range = num_samples * img_width + start_range + 1
  pixel_range = np.arange(start_range, end_range, img_width)
  sample_range_x = np.round(grid_x, 1)
  sample_range_y = np.round(grid_y, 1)
  plt.xticks(pixel_range, sample_range_x)
  plt.yticks(pixel_range, sample_range_y)
  plt.xlabel('z - dim 1')
  plt.ylabel('z - dim 2')
  
  fig_shape = np.shape(figure)
  if fig_shape[2] == 1:
    figure = figure.reshape((fig_shape[0], fig_shape[1]))
  
  plt.imshow(figure)
  plt.show()




# Main
data = (input_test, target_test)
a, b = visualize(encoder,decoder,data)
printa(b)
viz_latent_space(encoder, data)
viz_decoded(encoder, decoder, data)


