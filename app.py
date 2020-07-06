import os
from flask import Flask, render_template, request, redirect, url_for, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import *
from sklearn.svm import *
import pandas

app = Flask(__name__)
global Classifier
global Vectorizer

# load data

data = pandas.read_csv('/home/smsc/NLP/smsspamcollection/SMSSpamCollection', sep='\t', encoding='latin-1',
                       names=["label", "message"])
data.rename(columns={'label': 'v1', 'message': 'v2'}, inplace=True)
train_data = data[:4500]
test_data = data[4500:]

# train model
Classifier = OneVsRestClassifier(SVC(kernel='linear', probability=True))
Vectorizer = TfidfVectorizer()
vectorize_text = Vectorizer.fit_transform(train_data.v2)
Classifier.fit(vectorize_text, train_data.v1)


@app.route('/')
@app.route('/spam')
def spam_page():
    return render_template('spam.html')


@app.route('/results', methods=['POST'])
def results_page():
    if request.method == "POST":
        message = request.form['content']
        print(message)
        error = ''
        predict_proba = ''
        predict = ''

        global Classifier
        global Vectorizer
        try:
            if len(message) > 0:
                vectorize_message = Vectorizer.transform([message])
                predict = Classifier.predict(vectorize_message)[0]
                predict_proba = Classifier.predict_proba(vectorize_message).tolist()
                ham_data = predict_proba[0][0]
                ham_data = "{:.2f}".format(ham_data)
                spam_data = predict_proba[0][1]
                spam_data = "{:.2f}".format(spam_data)
        except BaseException as inst:
            error = str(type(inst).__name__) + ' ' + str(inst)

        if predict == "spam":
            return render_template('results.html', my_prediction=predict, spam_data=spam_data, ham_data=ham_data)
        elif float(ham_data) < 0.7:
            return render_template('results.html', my_prediction="spam", spam_data=spam_data)

        else:
            return render_template('results.html', my_prediction=predict, spam_data=spam_data, ham_data=ham_data)


if __name__ == "__main__":
    app.run(host="192.168.74.51", port=8002, debug=True)
