# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 16:46:23 2020

@author: a.ragab
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from keras.models import Sequential

from tensorflow import keras
from tensorflow.keras import backend as K

from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from os import listdir, makedirs

y=pd.read_csv('labels.csv')
Datat_sub=pd.read_csv('sample_submission.csv')

print(len(listdir(('train'))), len(y))
print(len(listdir(('test'))), len(Datat_sub))



selected_breed_list = list(y.groupby('breed').count().sort_values(by='id', ascending=False).head(120).index)
y = y[y['breed'].isin(selected_breed_list)]
y['target'] = 1
#y['rank'] = y.groupby('breed').rank()['id']
labels_pivot = y.pivot('id', 'breed', 'target').reset_index().fillna(0)
#np.random.seed(seed=1987)
rnd = np.random.random(len(labels))
train_idx = rnd < 0.8
valid_idx = rnd >= 0.8
y_train = y[selected_breed_list].values
ytr = y_train[train_idx]
yv = y_train[valid_idx]

