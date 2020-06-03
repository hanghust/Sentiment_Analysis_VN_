import numpy as np
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence
import pandas as pd 

# Xử lí bình luận
import re 
from pyvi import ViTokenizer



# print(file_crawl_txt)
def preprocessing_data(path,input_data, file_crawl_csv, file_crawl_txt):
    # load các từ dừng của tiếng việt 
    stopwords = []
    f = open('vietnamese-stopwords.txt', 'r', encoding="utf8")
    for line in f:
        line = line.rstrip()
        # print(line)
        line = line.replace(' ', '_')
        
        stopwords.append(line)
    f.close()

    #read data
    if path == '':
        data = input_data
    else:
        data = pd.read_csv(path + file_crawl_csv)
        data.head()
    dataset_cmt = data['Comment']
    data = np.array(data)
    # print(dataset_cmt)
    count_new_reviews = len(dataset_cmt)
    data_review = open(file_crawl_txt, "w+", encoding='utf8')
    for review in dataset_cmt:
        if str(review) == 'nan':
            continue
        data_review.write(str(review) +'\n')
    data_review.close()

    content = open(file_crawl_txt, 'r', encoding="utf8")
    data_prepared = open(path + 'review_preprocessing.txt', 'w+', encoding='utf8')
    for sentence in content: 
        
        # viết thường 
        sentence = sentence.lower()
        # bỏ các liên kết 
        sentence = re.sub("(http|https|ftp)[\n\S]+","",sentence)

        # loại bỏ kí tự đặc biệt và số
        sentence = re.sub("\W|\d" ," ", sentence) 

        # tách từ
        sentence  = ViTokenizer.tokenize(sentence)
        # print(content)              

        # print(content)
        text = ""
        # bỏ stopword
        sentence = sentence.split()
        for word in sentence: 
            if word not in stopwords:
                text += word + ' '
        data_prepared.write(text + "\n")
    data_prepared.close()
    content.close()
    return count_new_reviews, data
# path = "Data/data_crawl/tutorial/"
# file_preprocessing_txt = 'Data/data_crawl/tutorial/review_preprocessing.txt'
# file_crawl_csv = "data_crawl.csv"
# file_crawl_txt = path + "review.txt"
# count_new_reviews, data = preprocessing_data(path, file_crawl_csv, file_crawl_txt)
# data_ = np.array(data)
# for i in len(data_):
#     data_t = data_[i]
#     print("0", data_t[0])
#     print("1", data_t[1])
#     print("2", data_t[2])
