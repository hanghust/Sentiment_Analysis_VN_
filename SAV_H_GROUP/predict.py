# Code load model to test
from keras.models import load_model
from keras.models import model_from_json
from sklearn import metrics
from keras.optimizers import Adadelta
import numpy as np
import preprocessing
import prepare_text
# load json and create model

def predict_review(X_test):
    json_file = open('Model/model_tmp.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("Model/model_tmp.h5")
    adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-06, decay=1e-3)
    loaded_model.compile(loss='categorical_crossentropy',
              optimizer=adadelta,
              metrics=['accuracy'])
    print("Loaded model from disk")
    total_review = len(X_test)
    y_pred = loaded_model.predict(X_test)
    y_pred = np.array(y_pred).round()
    y_pred.astype(int)
    count_pos = 0
    count_neg = 0
    for elm in y_pred:
        if elm[0] == 1:
            count_pos +=1
            continue
        count_neg +=1
    return count_pos, count_neg, total_review

# path = "Data/data_crawl/tutorial/"
# file_preprocessing_txt = 'Data/data_crawl/tutorial/review_preprocessing.txt'
# file_crawl_csv = "data_crawl.csv"
# # print(path+file_crawl_csv)
# file_crawl_txt = path + "review.txt"
# count_new_reviews, data = preprocessing.preprocessing_data(path, '', file_crawl_csv, file_crawl_txt)
# number_word, review_data, review_data_np = prepare_text.prepare_data(path, file_preprocessing_txt)
# count_pos, count_neg, total_review = predict_review(review_data)
# for elm in review_data:
#     print(elm)