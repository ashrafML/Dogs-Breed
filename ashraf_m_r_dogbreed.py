# -*- coding: utf-8 -*-
"""
using transfer learing in pretrain model Xception
to predict dog breed
"""
import sys
import datetime
import numpy as np 
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from matplotlib import cm
from tensorflow.keras.layers import *
import matplotlib.pyplot as plt
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
plt.style.use('ggplot')
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
from keras.applications import xception


#using kaggle to download dataset
"""from google.colab import files
files.upload()
# Let's make sure the kaggle.json file is present.
!ls -lha kaggle.json
# Next, install the Kaggle API client.
!pip install -q kaggle

# The Kaggle API client expects this file to be in ~/.kaggle,
# so move it there.
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/

# This permissions change avoids a warning on Kaggle tool startup.
!chmod 600 ~/.kaggle/kaggle.json"""

from google.colab import drive
drive.mount('/content/gdrive')

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/gdrive/My Drive/DogBreeds

!unzip train.zip



"""#Task 1: Dog Breed"""

labels=pd.read_csv('labels.csv')

#load train
from os import listdir, makedirs
print(len(listdir(('train'))), len(labels))
#will load only 20 class 
selected_breed_list = list(labels.groupby('breed').count().sort_values(by='id', ascending=False).head(20).index)

labels = labels[labels['breed'].isin(selected_breed_list)]
labels['filename'] = labels.apply(lambda x: ('train/' + x['id'] + '.jpg'), axis=1)



#Will Process only 20 class so get only images that related 20 class
breeds = pd.Series(labels['breed'])
print("total number of breeds to classify",len(breeds.unique()))
label_enc = LabelEncoder()
np.random.seed(seed=42)
rnd = np.random.random(len(labels))
train_idx = rnd < 0.9
valid_idx = rnd >= 0.9
y_train = label_enc.fit_transform(labels["breed"].values)
ytr = y_train[train_idx]
yv = y_train[valid_idx]
im_size=299#image size in xception modal\
    
x_train = np.zeros((train_idx.sum(), im_size, im_size, 3), dtype='float32')
x_valid = np.zeros((valid_idx.sum(), im_size, im_size, 3), dtype='float32')
train_i = 0
valid_i = 0
#read image from source and resize
def read_img(img_id, train_or_test, size):
    """Read and resize image.
    # Arguments
        img_id: string
        train_or_test: string 'train' or 'test'.
        size: resize the original image.
    # Returns
        Image as numpy array.
    """
    path =  train_or_test + "/" + img_id + ".jpg"
    img = image.load_img((path), target_size=size)
    return image.img_to_array(img)
#process image in xception preprocessing
from keras.preprocessing import image
from tqdm import tqdm
for i, img_id in tqdm(enumerate(labels['id'])):
    img = read_img(img_id, 'train', (im_size, im_size))
    x = xception.preprocess_input(np.expand_dims(img.copy(), axis=0))
    if train_idx[i]:
        x_train[train_i] = x
        train_i += 1
    elif valid_idx[i]:
        x_valid[valid_i] = x
        valid_i += 1
print('Train Images shape: {} size: {:,}'.format(x_train.shape, x_train.size))

batch_size = 50
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rotation_range=45,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.25,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

train_generator = train_datagen.flow(x_train, 
                                     ytr, 
                                     batch_size=batch_size)


valid_datagen = ImageDataGenerator()

valid_generator = valid_datagen.flow(x_valid, 
                                     yv, 
                                     batch_size=batch_size)
#load load Xception pretrain model
from keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Dropout
from keras.optimizers import Adam, SGD, RMSprop
from keras.models import Model, Input
# create the base pre-trained model
base_model = xception.Xception(weights='imagenet', include_top=False)
# first: train only the top layers (which were randomly initialized)

#  freeze all convolutional Xception layers
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = BatchNormalization()(x)
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dropout(0.5)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
# and a logistic layer and set it to the number of breeds we want to classifiy, 
predictions = Dense(120, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

optimizer = tf.optimizers.RMSprop(lr=0.001, rho=0.9)
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=["accuracy"])

hist_dog_breed = model.fit_generator(train_generator,
                           steps_per_epoch=train_idx.sum() // batch_size,
                           epochs=10, 
                          
                           validation_data=valid_generator,
                           validation_steps=valid_idx.sum() // batch_size)

print(hist_dog_breed.history.keys())

# summarize history for accuracy
plt.plot(hist_dog_breed.history['accuracy'])
plt.plot(hist_dog_breed.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(hist_dog_breed.history['loss'])
plt.plot(hist_dog_breed.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

def predict_from_image(img_path):

    img = image.load_img(img_path, target_size=(299, 299))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]
    
    pred = model.predict(img_tensor)
    sorted_breeds_list = sorted(selected_breed_list)
    predicted_class = sorted_breeds_list[np.argmax(pred)]
    
    plt.imshow(img_tensor[0])                           
    plt.axis('off')
    plt.show()

    return predicted_class

# image path
!wget http://www.dogbreedslist.info/uploads/allimg/dog-pictures/Scottish-Deerhound-2.jpg

img_path = 'Scottish-Deerhound-2.jpg'    # dog
predict_from_image(img_path)







