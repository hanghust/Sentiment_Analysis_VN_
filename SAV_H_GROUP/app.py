#!/usr/bin/python
from flask import Flask, render_template, Response, json, request, session, jsonify, redirect, url_for
import preprocessing, prepare_text, predict
import pygal
from pygal.style import Style
import os
import pandas as pd
from keras.preprocessing import sequence as sq
app = Flask(__name__, static_url_path='/Static')

path = "Data/data_crawl/tutorial/"
file_preprocessing_txt = 'Data/data_crawl/tutorial/review_preprocessing.txt'
file_crawl_csv = "data_crawl.csv"
# print(path+file_crawl_csv)
file_crawl_txt = path + "review.txt"
count_new_reviews, data = preprocessing.preprocessing_data(path, '', file_crawl_csv, file_crawl_txt)
number_word, review_data, review_data_np = prepare_text.prepare_data(path, file_preprocessing_txt)
count_pos, count_neg, total_review = predict.predict_review(review_data)

@app.route('/')
def index():
    return redirect('/login.html')

@app.route('/index.html', methods = ['GET','POST'])
def home():

    # đồ thị biễu diễn độ tin cậy của mô hình

    figure_chart_acc = pygal.Bar()
    figure_chart_acc.title = 'Đồ thị biểu diễn độ tin cậy của mô hình SAV'
    figure_chart_acc.x_labels = map(str, ['Accuracy', 'Precision', 'Recall(macro)', 'Recall(micro)', 'F1-Score(macro)', 'F1-Score(micro)'])
    figure_chart_acc.add('%acc', [86.67, 86.67, 86.67, 86.67, 86.64011005181783, 86.67])
    figure_chart_acc_ = figure_chart_acc.render_data_uri()
    return render_template('index.html',
                           total_review = total_review,
                           number_word = number_word,
                        #    figure_chart = figure_chart_,
                           figure_chart_acc = figure_chart_acc_)


@app.route('/login.html')
def login():
    return render_template('login.html')

@app.route('/profile.html')
def profile():
    return render_template('profile.html')

@app.route('/analysis_preprocessing.html')
def analysis_preprocessing():
    number = []
    for elm in review_data_np:
        number.append(len(elm))
    figure_chart = pygal.Bar()
    figure_chart.title = 'Đồ thị phân bố độ dài của các review'
    figure_chart.x_labels = map(str, range(len(number)))
    figure_chart.add('reviews', number)
    figure_chart_ = figure_chart.render_data_uri()

    table_data = data
    len_data = len(table_data)
    return render_template('analysis_preprocessing.html',
        count_new_reviews = count_new_reviews,
        number_word = number_word,
        figure_chart = figure_chart_,
        table_data = table_data,
        len_data = len_data
    )

@app.route('/activity_log.html')
def activity_log():
    return render_template('activity_log.html',
        count_new_reviews = count_new_reviews,
        number_word = number_word,)

@app.route('/analysis_crawl.html', methods =['GET', 'POST'])
def analysis_crawl():
    if request.method == 'POST':
        myCmd = 'cd D:\\DSS\\doanIII\\doan_3\\SAV_H_GROUP\\Data\\data_crawl\\tutorial && D: && scrapy crawl crawler -o comments.csv'
        os.system(myCmd)
    return render_template('analysis_crawl.html')

@app.route('/register.html')
def register():
    return render_template('register.html')

@app.route('/forgot-password.html')
def forgot_password():
    return render_template('forgot-password.html')

@app.route('/charts.html')
def charts():
    pie_chart_pred = pygal.Pie()
    pie_chart_pred.title = 'Kết quả phân tích cảm xúc(in %)'
    pie_chart_pred.add('positive', count_pos*100/(count_pos+count_neg))
    pie_chart_pred.add('Negative', count_neg*100/(count_pos+count_neg))
    pie_chart_pred_ = pie_chart_pred.render_data_uri()

    number = []
    for elm in review_data_np:
        number.append(len(elm))
    figure_chart = pygal.Bar()
    figure_chart.title = 'Đồ thị phân bố độ dài của các review'
    figure_chart.x_labels = map(str, range(len(number)))
    figure_chart.add('reviews', number)
    figure_chart_ = figure_chart.render_data_uri()
 
    figure_chart_acc = pygal.Bar()
    figure_chart_acc.title = 'Đồ thị biểu diễn độ tin cậy của mô hình SAV'
    figure_chart_acc.x_labels = map(str, ['Accuracy', 'Precision', 'Recall(macro)', 'Recall(micro)', 'F1-Score(macro)', 'F1-Score(micro)'])
    figure_chart_acc.add('%acc', [86.67, 86.67, 86.67, 86.67, 86.64011005181783, 86.67])
    figure_chart_acc_ = figure_chart_acc.render_data_uri()

    return render_template('charts.html',
    pie_chart_pred = pie_chart_pred_,
    total_review = total_review,
    number_word = number_word,
    count_pos = count_pos,
    count_neg = count_neg,
    figure_chart = figure_chart_,
    figure_chart_acc = figure_chart_acc_
    )

@app.route('/tables.html')
def table():
    table_data = data
    len_data = len(table_data)
    return render_template('tables.html',
                            table_data = table_data,
                            len_data = len_data)
                            
@app.route('/404.html')
def error_show():
    return render_template('404.html')

@app.route('/data_3.html', methods = ['POST', 'GET'])
def chonse_data():
    data_ = []
    review_data_np_ = []
    count_pos_ = 0
    count_neg_ = 0
    total_review_ = 0
    input_data = []
    len_data = 0
    count_new_reviews_ =0
    number_word_ = 0
    path_ = ''
    file_preprocessing_txt = 'review_preprocessing.txt'
    # file_preprocessing_txt = path + file_preprocessing_txt
    if request.method == 'POST':
        user = request.files.get('csv')
        input_data= pd.read_csv(user)
        count_new_reviews_, data_ = preprocessing.preprocessing_data('', input_data, file_crawl_csv, file_crawl_txt)
        number_word_, review_data_, review_data_np_ = prepare_text.prepare_data(path, file_preprocessing_txt)
        count_pos_, count_neg_, total_review_ = predict.predict_review(review_data)
        # table_data = data_
        len_data = len(data_)

        pie_chart_pred = pygal.Pie()
        pie_chart_pred.title = 'Kết quả phân tích cảm xúc(in %)'
        pie_chart_pred.add('positive', count_pos_*100/(count_pos_+count_neg_))
        pie_chart_pred.add('Negative', count_neg_*100/(count_pos_+count_neg_))
        pie_chart_pred_ = pie_chart_pred.render_data_uri()

        number = []
        for elm in review_data_np_:
            number.append(len(elm))
        figure_chart = pygal.Bar()
        figure_chart.title = 'Đồ thị phân bố độ dài của các review'
        figure_chart.x_labels = map(str, range(len(number)))
        figure_chart.add('reviews', number)
        figure_chart_ = figure_chart.render_data_uri()
        return render_template('data_3.html',
                            table_data = input_data,
                            len_data = len_data,
                            pie_chart_pred = pie_chart_pred_,
                            figure_chart = figure_chart_)
    else:
         return render_template('data_3.html')

if __name__ == '__main__':
    app.run(debug=True, port=8000)