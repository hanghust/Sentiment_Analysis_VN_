import numpy as np
import re
import os
import pickle
import _pickle
# from __future__ import print_function
from socket import socket
from keras.layers.merge import concatenate
np.random.seed(1337)  # for reproducibility
from keras.preprocessing import sequence
from keras.models import Sequential
# from load_data import load_data_shuffle
from keras.preprocessing import sequence as sq
import pandas as pd
# import preprocessing 
def prepare_data(path, file_preprocessing_txt):
    word2id= {}
    id2word={}
    index = 1
    maxlen = 0
    avglen = 0
    count100 = 0
    save_file = path + 'review_preprocessing'

    # for open_file, save_file in zip(open_files,save_files):
    pos = []
    file = open(file_preprocessing_txt, 'r', encoding='utf8')
    for aline in file.readlines():
        aline = aline.replace('\n', "")
        ids = np.array([], dtype='int32')
        for word in aline.split(' '):
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
    np.save(save_file, pos)
    number_word = len(word2id)
    for li in pos:
        if maxlen < len(li):
            maxlen = len(li)
        avglen += len(li)
        if len(li) > 250:
            count100+=1
    review_data = path + "review_preprocessing.npy"
    np_load_old = np.load
    # np.load.__defaults__=(None, True, True, 'ASCII')
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True)
    review_data_np = np.load(review_data)
    # np.load.__defaults__=(None, False, True, 'ASCII')
    np_load = np_load_old
    review_data = sq.pad_sequences(review_data_np, maxlen=400)
    print(review_data.shape)
    total_review = len(review_data)
    return number_word, review_data, review_data_np


