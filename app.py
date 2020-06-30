from flask import Flask, request, redirect, render_template

app = Flask(__name__)

# importing the Dataset
import pandas as pd

messages = pd.read_csv('/home/smsc/NLP/smsspamcollection/SMSSpamCollection', sep='\t', names=["label", "message"])

# Data cleaning and preprocessing
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

ps = PorterStemmer()
wordnet = WordNetLemmatizer()
corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    review = review.lower()
    review = review.split()
    review = [wordnet.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# creating Tf-IDF model
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

cv = TfidfVectorizer(max_features=5000)
x = cv.fit_transform(corpus).toarray()

y = pd.get_dummies(messages['label'])
# taking spam column values only
y = y.iloc[:, 1].values

# Train test split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

# Training model using naive bayes classifier

from sklearn.naive_bayes import MultinomialNB

spam_detect_model = MultinomialNB().fit(x_train, y_train)
y_pred = spam_detect_model.predict(x_test)

# prediction with user text
clf = MultinomialNB()
clf.fit(x_train, y_train)
clf.score(x_test, y_test)


@app.route('/')
@app.route('/spam')
def spam_page():
    return render_template('spam.html')


@app.route('/results', methods=['POST'])
def results_page():
    if request.method == "POST":
        message = request.form['content']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
        return render_template('results.html', my_prediction=my_prediction)


if __name__=="__main__":
    app.run(port=8002, debug=True)
