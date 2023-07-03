from datetime import timedelta
from flask import Flask, render_template, request, session, redirect, url_for, jsonify
from flask_wtf import csrf
from flask_wtf.csrf import CSRFProtect
from models import db, User
from forms import SignupForm, LoginForm
import secrets
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import accuracy_score
from textstat import flesch_reading_ease
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
import string
import nltk
import requests
import joblib
from bs4 import BeautifulSoup
import newspaper

rfc = joblib.load('rf_trained')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

port_stem = PorterStemmer()
stop_words = stopwords.words('english')


def preprocessing(text):
  stemmed_content = re.sub('[^a-zA-Z]', ' ', text)
  stemmed_content = stemmed_content.lower()
  stemmed_content = stemmed_content.split()
  stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stop_words]
  stemmed_content = ' '.join(stemmed_content)
  return stemmed_content


def extract_custom_features(text):
  num_words = len(text.split())
  lexical_diversity = len(set(text.split())) / float(len(text.split()))
  sentiment_analyzer = SentimentIntensityAnalyzer()
  sentiment_scores = sentiment_analyzer.polarity_scores(text)
  sentiment = sentiment_scores['compound']
  reading_ease = flesch_reading_ease(text)
  url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
  query_params = {"query": text}
  headers = {"Authorization": "Bearer AIzaSyARLvA5Q934EHXzMBemFFAVkKLsq3CCNnw"}
  response = requests.get(url, params=query_params, headers=headers)
  response_json = response.json()
  if response_json.get("claims"):
    fact_check_rating = response_json["claims"][0]["claimReview"][0]["textualRating"]
    claim_text = response_json["claims"][0]["text"]
    fact_check_url = response_json["claims"][0]["claimReview"][0]["url"]
  else:
    fact_check_rating = None
    claim_text = None
    fact_check_url = None
  if fact_check_rating == None:
    fc = 0
  else:
    fc = 1
  return {'num_words': num_words,
          'lexical_diversity': lexical_diversity,
          'sentiment': sentiment,
          'reading_ease': reading_ease
          }


app = Flask(__name__)

# App configuration
app.config['DEBUG'] = True
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:postgresps@localhost/postgres'
app.config['SECRET_KEY'] = secrets.token_hex(16)
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30)
app.config['WTF_CSRF_SECRET_KEY'] = 'a secret key'

# CSRF protection
csrf = CSRFProtect(app)

# Initialize the database
db.init_app(app)

@app.route("/")
def index():
  return render_template("index.html")

@app.route("/about")
def about():
  return render_template("about.html")


@app.route("/signup", methods=["GET", "POST"])
@csrf.exempt
def signup():
  if 'email' in session:
    return redirect(url_for('home'))

  form = SignupForm()

  if request.method == "POST":
    if form.validate() == False:
      return render_template('signup.html', form=form)
    else:
      newuser = User(form.first_name.data, form.last_name.data, form.email.data, form.password.data)
      db.session.add(newuser)
      db.session.commit()

      session['email'] = newuser.email
      return redirect(url_for('home'))

  elif request.method == "GET":
    return render_template('signup.html', form=form)

@app.route("/login", methods=["GET", "POST"])
def login():
  if 'email' in session:
    return redirect(url_for('home'))

  form = LoginForm()

  if request.method == "POST":
    if form.validate() == False:
      return render_template("login.html", form=form)
    else:
      email = form.email.data
      password = form.password.data

      user = User.query.filter_by(email=email).first()
      if user is not None and user.check_password(password):
        session['email'] = form.email.data
        return redirect(url_for('home'))
      else:
        return redirect(url_for('login'))

  elif request.method == 'GET':
    return render_template('login.html', form=form)

@app.route("/logout")
def logout():
  session.pop('email', None)
  return redirect(url_for('index'))

@app.route("/home", methods=["GET", "POST"])
def home():
  if 'email' not in session:
    return redirect(url_for('login'))

  return render_template("home.html")

@app.route('/predict1')
def predict1():
    return render_template('predict1.html')

@app.route('/predict1', methods=['POST'])
def predict():
    url = request.form['url']
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    title_tag = soup.body.find(['h1', 'h2'])
    title = title_tag.get_text()
    article = newspaper.Article(url)
    article.download()
    article.parse()
    body_text = article.text
    body_text = title + body_text

    def remove_advertisement_and_also_read(news_content):
      pattern = re.compile(r'advertisement', re.IGNORECASE)
      while True:
        match = pattern.search(news_content)
        if match is None:
          break
        start_index = match.start()
        end_index = start_index
        for i in range(3):
          end_index = news_content.find('\n', end_index + 1)
          if end_index == -1:
            end_index = len(news_content)
            break
        news_content = news_content[:start_index] + news_content[end_index:]
      pattern = re.compile(r'^also read.*', re.IGNORECASE | re.MULTILINE)
      news_content = pattern.sub('', news_content)
      return news_content

    x = remove_advertisement_and_also_read(body_text)

    input_text = x
    preprocessed_text = preprocessing(input_text)
    custom_features = extract_custom_features(preprocessed_text)
    custom_features_df = pd.DataFrame(custom_features, index=[0])

    vectorizer = TfidfVectorizer(max_features=5000)
    vectorizer.fit([preprocessed_text])
    tfidf_features = vectorizer.transform([preprocessed_text])
    zeros_matrix = csr_matrix(np.zeros((tfidf_features.shape[0], 5000 - tfidf_features.shape[1])))
    tfidf_features = hstack([tfidf_features, zeros_matrix])
    tfidf_features_df = pd.DataFrame(tfidf_features.toarray())

    all_features_df = pd.concat([custom_features_df, tfidf_features_df], axis=1)

    all_features_df.columns = all_features_df.columns.astype(str)

    predicted_class = rfc.predict(all_features_df)

    if 'email' not in session:
      return redirect(url_for('login'))

    if predicted_class[0] == 1:
      prediction = "TRUE news."
    else:
      prediction = "FAKE news."

    return render_template('predict1.html', prediction_text=prediction)

if __name__ == "__main__":
  app.run(debug=True)  # Use gunicorn or uwsgi on
