# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 13:52:33 2020

@author: kshit
"""

# Import Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization

# Initializing the CNN
classifier = Sequential()

# Step 1 - CONVOLUTION
classifier.add(Convolution2D(32, (3, 3), input_shape = (64, 64, 3), activation='relu'))
# Step 2 - POOLING
classifier.add(MaxPooling2D(pool_size = (2,2)))

# Second layer
classifier.add(Convolution2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))
# Batch normalization is a technique for training very deep neural networks that standardizes the 
# inputs to a layer for each mini-batch. This has the effect of stabilizing the learning process 
# and dramatically reducing the number of training epochs required to train deep networks.
# classifier.add(BatchNormalization())

# Third layer
classifier.add(Convolution2D(64, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))
# classifier.add(BatchNormalization())

# STEP 3 - FLATTENING
classifier.add(Flatten())

# STEP 4 - FULL CONNECTION
classifier.add(Dense(activation='relu', units=4096))  #First Hidden layer
classifier.add(Dropout(0.5)) # To prevent overfitting
classifier.add(Dense(activation='relu', units=4096))  #Second Hidden layer

classifier.add(Dense(activation='softmax', units=4)) #Output layer

#Compiling the CNN
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#PART 2 - Fitting the CNN to the image
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True) 

training_set = train_datagen.flow_from_directory('dataset/training_set', 
                                                 target_size = (64,64),
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                              target_size = (64,64),
                                              class_mode = 'categorical')

early_stopping = EarlyStopping(monitor='val_loss', patience=20, min_delta=0, verbose=1, mode='auto')

hist = classifier.fit_generator(training_set, 
                         samples_per_epoch = 960,
                         epochs = 25,
                         validation_data = test_set,
                         nb_val_samples = 240,
                         shuffle = True,
                         callbacks=[early_stopping])


classifier.evaluate_generator(test_set, 32)

import matplotlib.pyplot as plt
plt.plot(hist.history["accuracy"])
plt.plot(hist.history['val_accuracy'])
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title("model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy","Validation Accuracy","loss","Validation Loss"])
plt.show()


import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np
# Predict the values from the validation dataset

loss, accuracy = classifier.evaluate_generator(test_set)
print("Test: accuracy = %f  ;  loss = %f " % (accuracy, loss))
y_pred =  classifier.predict_generator(test_set)
y_p = np.where(y_pred > 0.5, 1,0)
test_data=test_set.unbatch()
y_g=[]
for image, label in  test_data:
  y_g.append(label.numpy())

confusion_mtx = confusion_matrix(y_g, y_p) 
# plot the confusion matrix
f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Blues",linecolor="gray", fmt= '.1f',ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# import matplotlib.pyplot as plt
# import itertools 
# import numpy as np
# from sklearn.metrics import classification_report, confusion_matrix


# def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
#     plt.figure(figsize=(10,10))

#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()

#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)

#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         cm = np.around(cm, decimals=2)
#         cm[np.isnan(cm)] = 0.0
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, cm[i, j],
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")
#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
    
# target_names = []
# for key in training_set.class_indices:
#     target_names.append(key)
    
# print(target_names)


# Y_pred = classifier.predict_generator(test_set)
# y_pred = np.argmax(Y_pred, axis=1)
# print('Confusion Matrix')
# cm = confusion_matrix(test_set.classes, y_pred)
# plot_confusion_matrix(cm, target_names, title='Confusion Matrix')

# ACCURACY OBTAINED - 72.81% ///// VALIDATION ACCURACY LOW - 42.04%

#####################################################################################################
#####################################################################################################
#####################################################################################################


####################### BELOW CODE IS PURELY EXPERIMENTAL AS OF NOW ###########################

# IMPLEMENTING TRANSFER LEARNING USING VGG16 CNN ARCHITECTURE TO IMPROVE ACCURACY

from tensorflow.keras.applications import VGG16
import numpy as np

pre_trained_model = VGG16(include_top=False, weights='imagenet')
# vgg.summary()

for layer in pre_trained_model.layers[:15]:
    layer.trainable = False

for layer in pre_trained_model.layers[15:]:
    layer.trainable = True

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()    
from tensorflow.keras import layers
    

last_layer = pre_trained_model.get_layer('block5_pool')
last_output = last_layer.output
x = layers.Flatten()(last_output)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(4, activation='sigmoid')(x)

vggmodel = tf.keras.models.Model(pre_trained_model.input, x)

vggmodel.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

vggmodel.summary()

# from keras.utils import to_categorical

# vgg_features_train = vgg.predict(training_set)
# vgg_features_val = vgg.predict(test_set)

# train_target = to_categorical(training_set.labels)
# val_target = to_categorical(test_set.labels)


# model2 = Sequential()
# # model2.add(vgg)
# model2.add(Flatten())
# model2.add(Dense(4096, activation='relu'))
# model2.add(Dropout(0.5))
# # model2.add(Dense(128, activation='relu'))

# # # model2.add(BatchNormalization())
# model2.add(Dense(4, activation='softmax'))

# # compile the model
# model2.compile(optimizer='adam', metrics=['accuracy'], loss='categorical_crossentropy')

# # train model using features generated from VGG16 model
# hist = model2.fit(vgg_features_train, train_target, epochs=100, batch_size=32, validation_data=(vgg_features_val, val_target))

import pandas as pd
import IPython.display as display
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os,random
import matplotlib.pyplot as plt


def My_CNNmodel():

  model = tf.keras.models.Sequential()
  model.add(layers.Conv2D(8, (3, 3), padding='same',activation='relu' ))
  model.add(layers.MaxPooling2D(pool_size=(2,2)))
  model.add(layers.Conv2D(16, (3, 3), padding='same',activation='relu'))
  model.add(layers.MaxPooling2D(pool_size=(2,2)))
  model.add(layers.Conv2D(32, (3, 3), padding='same',activation='relu'))
  model.add(layers.MaxPooling2D(pool_size=(2,2)))
  model.add(layers.Conv2D(64, (3, 3), padding='same',activation='relu'))
  model.add(layers.MaxPooling2D(pool_size=(2,2)))
  model.add(layers.Flatten())
  model.add(layers.Dense(512, activation='relu'))
  model.add(layers.Dense(4, activation='softmax'))

  opt=tf.keras.optimizers.Adam(0.001)
  model.compile(optimizer=opt,
              loss='categorical_crossentropy', # loss='categorical_crossentropy' if softmax
              metrics=['accuracy'])

  return model

model=My_CNNmodel()
model.summary()

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True) 

dataset_train = train_datagen.flow_from_directory('dataset/training_set', 
                                                 target_size = (64,64),
                                                 class_mode = 'categorical')

dataset_test = test_datagen.flow_from_directory('dataset/test_set',
                                              target_size = (64,64),
                                              class_mode = 'categorical')

hist=model.fit_generator(dataset_train,epochs=20,validation_data=dataset_test)

def plot_model_history(model_history, acc='accuracy', val_acc='val_accuracy'):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    axs[0].plot(range(1,len(model_history.history[acc])+1),model_history.history[acc])
    axs[0].plot(range(1,len(model_history.history[val_acc])+1),model_history.history[val_acc])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history[acc])+1),len(model_history.history[acc])/10)
    axs[0].legend(['train', 'val'], loc='best')
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()
    
plot_model_history(hist)

from tensorflow.keras.applications import VGG16

pre_trained_model = VGG16(include_top=False, weights="imagenet")
    
for layer in pre_trained_model.layers[:15]:
    layer.trainable = False

for layer in pre_trained_model.layers[15:]:
    layer.trainable = True
    
last_layer = pre_trained_model.get_layer('block5_pool')
last_output = last_layer.output
x = layers.Flatten()(last_output)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(4, activation='softmax')(x)

vggmodel = tf.keras.models.Model(pre_trained_model.input, x)

vggmodel.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

vggmodel.summary()
