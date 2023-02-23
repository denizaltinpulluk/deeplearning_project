# -*- coding: utf-8 -*-
"""
Deep Learning Class final Project
"""
from scipy.io import wavfile
from scipy import signal
import numpy as np
import tensorflow.keras as K
#import matplotlib.pyplot as plt
"""
Import sound file to numpy array(1D)
"""
rate1, data1 = wavfile.read("Iphone_SE_Train.wav") 
rate2, data2 = wavfile.read("Iphone5_Train.wav") 
rate3, data3 = wavfile.read("Iphone6_Train.wav") 
rate4, data4 = wavfile.read("SamSung_train.wav") 

_, test1 = wavfile.read("Iphone5_test.wav") 
_, test2 = wavfile.read("Iphone6_Test.wav") 
_, test3 = wavfile.read("SamSung_test.wav") 
_,background=wavfile.read("Background_Noise.wav") 

"""
Prepare the training data Array with data augumentation
"""

"""
Reshape the data array
"""
data1=data1[0:158760000]
data1=data1.reshape(360,441000)

data2=data2[0:158760000]
data2=data2.reshape(360,441000)

data3=data3[0:158760000]
data3=data3.reshape(360,441000)

data4=data4[0:158760000]
data4=data4.reshape(360,441000)

test1=test1[0:44100000]
test1=test1.reshape(100,441000)

test2=test2[0:44100000]
test2=test2.reshape(100,441000)

test3=test3[0:44100000]
test3=test3.reshape(100,441000)


background=background[0:5632000]
background=background.reshape(2000,2816)

"""
Extract train/test data from silence
"""
train=np.empty((1440,1408),dtype='int16')
train2=np.empty((1440,2816),dtype='int16')
train3=np.empty((1440,2816),dtype='int16')
test=np.empty((300,1408),dtype='int16')

train_1=np.concatenate((data1, data2,data3,data4), axis=0) # Merge datas
test_1=np.concatenate((test1, test2,test3), axis=0)

a1=np.absolute(train_1)
a2=np.argmax(a1, axis=1)
index_L=a2-441 # Front 10ms
index_R=a2+967 # Later 20ms+

index_L2=a2-882 # Front 20ms
index_R2=a2+1934 # Later 40ms+

index_L3=a2-1408 # Front 30ms+
index_R3=a2+1408 # Later 30ms+

b1=np.absolute(test_1)
b2=np.argmax(b1, axis=1)
indext_L=b2-441 # Front 10ms
indext_R=b2+967 # Later 20ms+

for i in range(0,1440):
    train[i]=train_1[i,index_L[i]:index_R[i]]
    train2[i]=train_1[i,index_L2[i]:index_R2[i]]
    train3[i]=train_1[i,index_L3[i]:index_R3[i]]
    
for j in range(0,300):   
    test[j]=test_1[j,indext_L[j]:indext_R[j]]


"""
Prepare Label
"""
y_train=np.empty((5820,),dtype='float32')
y_test=np.empty((600,),dtype='float32')
"""
STFT processing using scipy
"""
"""
Train True Sample
"""
f, t, Z = signal.stft(train, fs=44100, window='hann', nperseg=128, noverlap=64)
f1, t1, Z1 = signal.stft(train2, fs=44100, window='hann', nperseg=128, noverlap=0)
f2, t2, Z2 = signal.stft(train3, fs=44100, window='hann', nperseg=128, noverlap=0)
Z=np.abs(Z)
Z1=np.abs(Z1)
Z2=np.abs(Z2)
#plt.pcolormesh(t, f, np.abs(Zxx[2]),) #plot to check the STFT
"""
Background Sample
"""
f3, t3, Z3 = signal.stft(background, fs=44100, window='hann', nperseg=128, noverlap=0)
Z3=np.abs(Z3)
"""
Test True Sample
"""
f4, t4, Z4 = signal.stft(test, fs=44100, window='hann', nperseg=128, noverlap=64)
Z4=np.abs(Z4)
"""
Mix background with True data
"""
x_train=np.concatenate((Z, Z1, Z2, Z3[0:1500] ), axis=0)
x_test=np.concatenate((Z4, Z3[1500:1800] ), axis=0)
y_train[0:4320]=1
y_train[4320:5820]=0
y_test[0:300]=1
y_test[300:600]=0
'''
print(np.shape(x_train))
x_train= np.repeat(x_train[:,:,  np.newaxis], 32, axis=2)
x_test= np.repeat(x_test[:,:,  np.newaxis], 32, axis=2)
print(np.shape(x_train))
'''

x_train=x_train.reshape(5820,1495)
x_test=x_test.reshape(600,1495)
total_x=np.concatenate((x_train,x_test),axis=0)
print('------------------------------------------')

#y_train = K.utils.to_categorical(y_train, num_classes=2, dtype='float32')
#y_test = K.utils.to_categorical(y_test, num_classes=2, dtype='float32')
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
total_x = normalize(total_x, axis=0, norm='max')
pca = PCA(n_components=1024)
principalComponents = pca.fit_transform(total_x)
print(np.shape(principalComponents))
principalComponents=principalComponents.reshape(6420,32,32)
principalComponents= np.repeat(principalComponents[:,:,:,  np.newaxis], 3, axis=3)
x_train=principalComponents[0:5820]
x_test=principalComponents[5820:]


# y_train=y_train.reshape(5820,1)
# y_test=y_test.reshape(600,1)
del data1,data2,data3,data4,rate1,rate2,rate3,rate4,test,test1,test2,test3,train,train2,train3


from tensorflow.keras import backend as K
from tensorflow.keras.models import Model ,load_model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras import models
from keras import layers
from keras.applications.resnet50 import ResNet50

import numpy as np
import soundfile as sf
from keras import models
from keras import layers
from keras import optimizers
from keras.applications import VGG16

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(32, 32, 3))

for layer in conv_base.layers[:-4]:
    layer.trainable = False

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(2048, activation='tanh'))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              
              optimizer='adam',
              metrics=['binary_accuracy'])

history=model.fit(x_train, y_train, epochs=20, batch_size=36, validation_data=(x_test, y_test))
import matplotlib.pyplot as plt
plt.plot(history.history['binary_accuracy'])
plt.plot(history.history['val_binary_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
score=model.evaluate(x_test, y_test)
print(score[1])

