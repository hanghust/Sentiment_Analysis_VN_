import numpy as np
import re
from keras.preprocessing import sequence as sq
import pandas as pd 
from pyvi import ViTokenizer
from keras.callbacks import LearningRateScheduler
from keras.callbacks import TensorBoard
from gensim.models import FastText
from keras.layers.merge import concatenate
np.random.seed(1337)  # for reproducibility
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Lambda
# from keras.layers import Embedding
from keras.layers import Convolution1D, LSTM
from keras import backend as K
from keras.optimizers import Adadelta
from keras.layers import Dense, Dropout, Activation, Lambda,merge,Input,TimeDistributed,Flatten
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import scale

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
model_fasttext = FastText.load('Model/fasttext_gensim.model')

max_features = 41375 #21540#14300 
batch_size = 512
embedding_dims = 150
nb_filter = 150
filter_length = 3
hidden_dims = 100
nb_epoch =45
# log_dir="/content/drive/My Drive/Do_An_III/Sentiment_Analysis_Vietnamese/logs"
# logdir = "/content/drive/My Drive/Do_An_III/Sentiment_Analysis_Vietnamese/logs_dir/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = TensorBoard(log_dir=logdir)
f = open('Data/training_data/train_preprocessing.txt', 'w+', encoding="utf8")
f.truncate()
f.close()
f = open('Data/testing_data/test_preprocessing.txt', 'w+', encoding="utf8")
f.truncate()
f.close()
f = open('Data/validation_data/val_preprocessing.txt', 'w+', encoding="utf8")
f.truncate()
f.close()
f = open('Data/preprocessing.txt', 'w+', encoding="utf8")
f.truncate()
f.close()
"""Tiền xử lý dữ liệu:
1.   loại bỏ stopword, các ký tự đặc biệt
2.   viết thường toàn bộ từ, tách từ
"""
def preprocessing_data(path, file_comment_txt, file_split_txt, lable_):
    content = open(path+file_comment_txt, 'r', encoding="utf8")
    # data_prepared = open(file_preprocessing_txt, 'w+', encoding='utf8')
    data_prepared_train = open(path + file_split_txt, 'a+', encoding='utf8')
    data_preprocessing = open('Data/preprocessing.txt', 'a+', encoding='utf8')  
    label = []
    for sentence in content: 
        label.append(lable_)
        data_prepared_train.writelines(sentence)
        data_preprocessing.writelines(sentence)
    data_prepared_train.close()
    data_preprocessing.close()
    content.close()
    return label

path_train = 'Data/training_data/'
path_test = 'Data/testing_data/'
path_val = 'Data/validation_data/'
lable_train_pos = preprocessing_data(path_train, 'train_pos.txt', 'train_preprocessing.txt', [1, 0])
lable_train_neg = preprocessing_data(path_train, 'train_neg.txt', 'train_preprocessing.txt', [0, 1])
lable_test_neg = preprocessing_data(path_test, 'test_neg.txt', 'test_preprocessing.txt', [0, 1])
lable_pos_neg = preprocessing_data(path_test, 'test_pos.txt', 'test_preprocessing.txt', [1, 0])
lable_val_neg = preprocessing_data(path_val, 'val_neg.txt', 'val_preprocessing.txt', [0, 1])
lable_val_pos = preprocessing_data(path_val, 'val_pos.txt', 'val_preprocessing.txt', [1, 0])
lable_train_pos = np.array(lable_train_pos)
lable_train_neg = np.array(lable_train_neg)
lable_train = np.concatenate((lable_train_pos, lable_train_neg), axis=0)
print(lable_train.shape)
lable_val_pos = np.array(lable_val_pos)
lable_val_neg = np.array(lable_val_neg)
lable_val = np.concatenate((lable_val_pos, lable_val_neg), axis = 0)
print(lable_val.shape)

lable_test_pos = np.array(lable_pos_neg)
lable_test_neg = np.array(lable_test_neg)
lable_test = np.concatenate((lable_test_pos, lable_test_neg), axis=0)
print(lable_test.shape)

def vector_sentence(path_file_comment):
    data = []
    sents = open(path_file_comment, 'r', encoding='utf8').readlines()
    for sent in sents:
        data.append(sent.split())
    return data
path_file = 'Data/training_data/train_preprocessing.txt'
sentence_train = vector_sentence(path_file)


sentence_test = vector_sentence("Data/testing_data/test_preprocessing.txt")
sentence_val = vector_sentence("Data/validation_data/val_preprocessing.txt")

print('building tf-idf matrix ...')
def vector_sentence_tfidf(path_file_comment):
    data = []
    sents = open(path_file_comment, 'r', encoding='utf8').readlines()
    for sent in sents:
            # print(word)
        data.append(sent)
    return data
tfidf_text = vector_sentence_tfidf('Data/preprocessing.txt')
# print(tfidf_text[0])
vectorizer = TfidfVectorizer(min_df = 0.0005, max_df = 0.990)
matrix = vectorizer.fit_transform(tfidf_text)
tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
print('vocab size :', len(tfidf))
print(len(vectorizer.idf_))
# ####################################
#  build model word embedding with fasttext model in gensim
# # data = vector_sentence('Data/preprocessing.txt')
# # model_fasttext = FastText(size=150, window=10, min_count=1, workers=4, sg=1)
# # model_fasttext.build_vocab(data)
# # model_fasttext.train(data, total_examples = model_fasttext.corpus_count, epochs=model_fasttext.iter)
# # model_fasttext.save("Model/fasttext_gensim.model")

# # ############################
# # print(vectorizer.idf_)
def buildWordVector(sent, size):
    # for sent in tokens:
    vec = np.zeros(size).reshape((1, size))
    count = 0
    for word in sent:
        try:
            try: 
                # print("@@@@")
                vec += model_fasttext[word].reshape((1, size))* tfidf[word]
                count += 1 
            except KeyError:
                vec += model_fasttext[word].reshape((1, size))
                count += 1 
        except KeyError:
            try:
                # print("%%%%%")
                model_fasttext.build_vocab([sent], update=True)
                model_fasttext.train(sent, total_examples=len(sent), epochs=model_fasttext.iter)
                vec += model_fasttext[word].reshape((1, size))* tfidf[word]
                count += 1
            except KeyError:
                # print("^^^^^")
                model_fasttext.build_vocab([sent], update=True)
                model_fasttext.train(sent, total_examples=len(sent), epochs=model_fasttext.iter)
                vec += model_fasttext[word].reshape((1, size))
                count += 1
    if count != 0:
        vec /= count
    return vec


# # # # #########################################
# train_vecs_fasttext =  buildWordVector(sentence_train[0], 150)
# for sent in sentence_train:
#     # print(buildWordVector(sent, 150))
#     vecs = buildWordVector(sent, 150)
#     train_vecs_fasttext = np.concatenate(train_vecs_fasttext, vecs)
# print(train_vecs_fasttext.shape)
# train_vecs_fasttext = scale(train_vecs_fasttext[1:])

# val_vecs_fasttext = buildWordVector(sentence_val[0], 150)
# for sent in sentence_val:
#     vecs = buildWordVector(sent, 150)
#     val_vecs_fasttext = np.concatenate(val_vecs_fasttext, vecs)
# print(val_vecs_fasttext.shape)
# val_vecs_fasttext = scale(val_vecs_fasttext[1:])

# test_vecs_fasttext = buildWordVector(sentence_test[0], 150)
# for  sent in sentence_test:
#     vecs = buildWordVector(sent, 150)
#     test_vecs_fasttext = np.concatenate(val_vecs_fasttext[vecs])
# print(test_vecs_fasttext.shape)
# test_vecs_fasttext = scale(test_vecs_fasttext[1:])
# # #########################################

train_vecs_fasttext =  []
for sent in sentence_train:
    # print(buildWordVector(sent, 150))
    vecs = buildWordVector(sent, 150)
    train_vecs_fasttext.append(vecs)
train_vecs_fasttext = (np.array(train_vecs_fasttext))#.reshape(30000,150)
print(train_vecs_fasttext.shape)
# train_vecs_fasttext = scale(train_vecs_fasttext)

val_vecs_fasttext = []
for sent in sentence_val:
    vecs = buildWordVector(sent, 150)
    val_vecs_fasttext.append(vecs)
val_vecs_fasttext = (np.array(val_vecs_fasttext))#.reshape(10000,150)
print(val_vecs_fasttext.shape)
# val_vecs_fasttext = scale(val_vecs_fasttext)

test_vecs_fasttext = []
for  sent in sentence_test:
    vecs = buildWordVector(sent, 150)
    test_vecs_fasttext.append(vecs)
test_vecs_fasttext = (np.array(test_vecs_fasttext))#.reshape(10000, 150)
print(test_vecs_fasttext.shape)
# test_vecs_fasttext = scale(test_vecs_fasttext)
#################################################

print('Build model...')
model = Sequential()

input_layer = Input(shape=(None,150),dtype='float', name='main_input')
print(input_layer)               
def max_1d(X):
    return K.max(X, axis=1)

# we add a Convolution1D, which will learn nb_filter
# word group filters of size 3:

con3_layer = Convolution1D(nb_filter=nb_filter,
                    filter_length=3,
                    border_mode='valid',
                    activation='relu',
                    subsample_length=1,
                    batch_size = 128)(input_layer)

pool_con3_layer = Lambda(max_1d, output_shape=(nb_filter,))(con3_layer)

# we add a Convolution1D, which will learn nb_filter
# word group filters of size 4:

con4_layer = Convolution1D(nb_filter=nb_filter,
                    filter_length=5,
                    border_mode='valid',
                    activation='relu',
                    subsample_length=1)(input_layer)

pool_con4_layer = Lambda(max_1d, output_shape=(nb_filter,))(con4_layer)


# # we add a Convolution1D, which will learn nb_filter
# # word group filters of size 5:

con5_layer = Convolution1D(nb_filter=nb_filter,
                    filter_length=7,
                    border_mode='valid',
                    activation='relu',
                    subsample_length=1)(input_layer)

pool_con5_layer = Lambda(max_1d, output_shape=(nb_filter,))(con5_layer)


cnn_layer = concatenate([pool_con3_layer, pool_con4_layer,pool_con5_layer])
# # cnn_layer = concatenate([pool_con3_layer,pool_con4_layer])
# # cnn_layer = pool_con3_layer

# #LSTM


x = input_layer
lstm_layer = LSTM(64)(x)

cnn_lstm_layer = concatenate([lstm_layer, cnn_layer])
dense_layer = Dense(hidden_dims*2, activation='sigmoid')(cnn_lstm_layer)
output_layer= Dropout(0.2)(dense_layer)
output_layer = Dense(2, trainable=True,activation='softmax')(output_layer)




model = Model(input=[input_layer], output=[output_layer])
adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-06, decay= 1e-1)


model.compile(loss='categorical_crossentropy',
              optimizer=adadelta,
              metrics=['accuracy'])
model.summary()

          # callbacks=[LearningRateScheduler(lr_schedule,verbose=1), checkpoint])

checkpoint = ModelCheckpoint('Model/CNN-LSTM-weights/weights.h5',
                              monitor='val_acc', verbose=0, save_best_only=True,
                              mode='max')


history = model.fit(train_vecs_fasttext, lable_train, batch_size=batch_size,\
          # steps_per_epoch=len(X_train) / batch_size,\
          # validation_steps = len(X_train)/batch_size,
          epochs=nb_epoch,\
          callbacks=[checkpoint],\
          validation_data=(val_vecs_fasttext, lable_val))
# # #################################
