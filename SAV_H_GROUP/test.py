
"""1.   Tiền xử lý dữ liệu sau đó lưu dữ liệu vào file .npy tương ứng"""
from __future__ import print_function
from socket import socket
from keras.layers.merge import concatenate

np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Lambda
from keras.layers import Embedding
from keras.layers import Convolution1D, LSTM
from keras.datasets import imdb
from keras import backend as K
from keras.optimizers import Adadelta
# from load_data import load_data_shuffle
from keras.preprocessing import sequence as sq
from keras.layers import Dense, Dropout, Activation, Lambda,merge,Input,TimeDistributed,Flatten
from keras.models import Model
from keras.callbacks import ModelCheckpoint

import keras.backend.tensorflow_backend as K
from keras.callbacks import LearningRateScheduler
import numpy as np
import re
from gensim.models import FastText

word2id= {}
id2word={}
index = 1
avglen = 0
count100 = 0
max_features = 41375 #21540#14300 
maxlen = 400
batch_size = 1000
embedding_dims = 150
nb_filter = 150
filter_length = 3
hidden_dims = 100
#read file
train_pos_file = "Data/training_data/train_pos.txt"
train_neg_file = "Data/training_data/train_neg.txt"

test_pos_file = "Data/testing_data/test_pos.txt"
test_neg_file = "Data/testing_data/test_neg.txt"

val_pos_file = "Data/validation_data/val_pos.txt"
val_neg_file = "Data/validation_data/val_neg.txt"


open_files = [train_pos_file, train_neg_file, test_pos_file, test_neg_file, val_pos_file, val_neg_file]

#save file
train_pos_save = "Data/training_data/train_pos"
train_neg_save = "Data/training_data/train_neg"

test_pos_save = "Data/testing_data/test_pos"
test_neg_save = "Data/testing_data/test_neg"

val_pos_save = "Data/validation_data/val_pos"
val_neg_save = "Data/validation_data/val_neg"

save_files = [train_pos_save, train_neg_save, test_pos_save, test_neg_save, val_pos_save, val_neg_save]

for open_file, save_file in zip(open_files,save_files):
  pos = []
  file = open(open_file, 'r')

  for aline in file.readlines():
      aline = aline.replace('\n', "") 
      ids = np.array([], dtype='int32')
      for word in aline.split(' '):
          word = word.lower()
          if word in word2id:
              ids = np.append(ids, word2id[word])
          else:
              if word != '':
                  # print (word, "not in vocalbulary")
                  word2id[word] = index
                  id2word[index] = word
                  ids = np.append(ids, index)
                  index = index + 1
      if len(ids) > 0:
          pos.append(ids)

  file.close()
  print(len(pos))
  np.save(save_file, pos)
  for li in pos:
      if maxlen < len(li):
          maxlen = len(li)
      avglen += len(li)
      if len(li) > 400:
          count100+=1

import os
import pickle
import _pickle

"""1.   Gán lable tương ứng cho từng câu tương ứng đã qua bước tiền xử lý :


*   label = [1, 0] => tương ứng với câu tích cực
*   label = [0, 1] => tương ứng với câu tiêu cực
"""

def load_data_shuffle():

    train_pos_save = "Data/training_data/train_pos.npy"
    train_neg_save = "Data/training_data/train_neg.npy"

    test_pos_save = "Data/testing_data/test_pos.npy"
    test_neg_save = "Data/testing_data/test_neg.npy"

    val_pos_save = "Data/validation_data/val_pos.npy"
    val_neg_save = "Data/validation_data/val_neg.npy"
    # save np.load
    np_load_old = np.load

    # modify the default parameters of np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
    #Load train data
    pos_train = np.load(train_pos_save)
    neg_train = np.load(train_neg_save)

    y_pos_train = []
    for i in pos_train:
        y_pos_train.append([1,0])
    y_pos_train = np.array(y_pos_train)

    y_neg_train = []
    for i in neg_train:
        y_neg_train.append([0, 1])
    y_neg_train = np.array(y_neg_train)



    #load test data
    pos_test = np.load(test_pos_save)
    neg_test = np.load(test_neg_save)

    y_pos_test = []
    for i in pos_test:
        y_pos_test.append([1,0])
    y_pos_test = np.array(y_pos_test)

    y_neg_test = []
    for i in neg_test:
        y_neg_test.append([0, 1])
    y_neg_test = np.array(y_neg_test)


        #load val data
    pos_val = np.load(val_pos_save)
    neg_val = np.load(val_neg_save)

    y_pos_val = []
    for i in pos_val:
        y_pos_val.append([1,0])
    y_pos_val = np.array(y_pos_val)

    y_neg_val = []
    for i in neg_val:
        y_neg_val.append([0, 1])
    y_neg_val = np.array(y_neg_val)

    # restore np.load for future normal usage
    np.load = np_load_old

    X_train = np.concatenate([pos_train, neg_train])
    y_train = np.concatenate([y_pos_train, y_neg_train])

    X_val = np.concatenate([pos_val, neg_val])
    y_val = np.concatenate([y_pos_val, y_neg_val])

    X_test = np.concatenate([pos_test, neg_test])
    y_test = np.concatenate([y_pos_test, y_neg_test])

    print (X_train.shape, y_train.shape)
    print (X_test.shape, y_test.shape)
    print (X_val.shape, y_val.shape)

    return X_train, y_train, X_test, y_test, X_val, y_val
print('Loading data ...')

X_train, y_train, X_test, y_test, X_val, y_val = load_data_shuffle()
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')
print(len(X_val), 'val sequences')

max_features = 41375 #21540#14300 
maxlen = 400
batch_size = 1000
embedding_dims = 150
nb_filter = 150
filter_length = 3
hidden_dims = 100
X_train = sq.pad_sequences(X_train, maxlen=maxlen)
X_test = sq.pad_sequences(X_test, maxlen=maxlen)
X_val = sq.pad_sequences(X_val, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_val shape:', X_val.shape)
print('X_test shape:', X_test.shape)