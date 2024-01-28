from flask import Flask, render_template, request, redirect, url_for,jsonify
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from flask_sqlalchemy import SQLAlchemy
import nltk
from nltk import WordNetLemmatizer
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import xgboost
from xgboost import XGBClassifier
from joblib import dump, load
import joblib


import csv
import pandas as pd
import numpy as np
import re
import requests



def remove_punct(text):
    if isinstance(text, str):
        return "".join([ch for ch in text if ch not in string.punctuation])
    else:
        return str(text)


def tokenize(text):
    text = re.split('\s+' ,text)
    return [x.lower() for x in text]


nltk.download('stopwords')

def remove_stopwords(text):
    return [word for word in text if word not in nltk.corpus.stopwords.words('english')]


nltk.download('wordnet')
def lemmatize(text):
    word_net = WordNetLemmatizer()
    return [word_net.lemmatize(word) for word in text]


def return_sentences(tokens):
    return " ".join([word for word in tokens])


rf= joblib.load("random_forest_model.joblib")




app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI']='sqlite:///user_dark_pattern.db'
db=SQLAlchemy(app)


class user(db.Model):
    id= db.Column(db.Integer,primary_key=True)
    first_name=db.Column(db.String)
    last_name=db.Column(db.String)
    email= db.Column(db.String)
    mobile= db.Column(db.String)

    def __repr__(self):
        return '<Task %r' % self.id


url=""
@app.route('/receive_url', methods=['POST'])
def receive_url():
    data = request.json
    url = data.get('url')

def process_link(link):
    options = Options()
    options.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(link)
    return driver

def is_url_scrapable(url):
    try:
        response = requests.get(url)
        response.raise_for_status()

        if response.status_code == 200:
            return True
        else:
            return False

    except requests.exceptions.RequestException as e:
        return False

def extract_text(driver):
    i = 0
    extracted_text = ''
    while i < 5:
        extracted_text += driver.find_element("xpath","/html/body").text + '\n'
        driver.execute_script("window.scrollTo(0,document.body.scrollHeight)")
        i += 1
    return extracted_text

def splitter(input_string):
    lines = input_string.split("\n")
    csv_data = []

    for line in lines:
        parts = re.split(r'(?<!\d)\.(?!\d)', line)
        row = [part.strip() for part in parts if part.strip()]
        if row:
            csv_data.append(row)
    
    return csv_data
def has_letters(text):
    return bool(re.search('[a-zA-Z]', text))
def clean_csv():
    df = pd.read_csv("output.csv")
    column_name = df.columns[0]
    df = df[df[column_name].apply(has_letters)]

    # Optionally, you can reset the index if you want consecutive row numbers
    df.reset_index(drop=True, inplace=True)

    # print(df)

    return df
    

def array_to_csv(csv_data, filename):
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        for row in csv_data:
            csv_writer.writerow([row])


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/analysis', methods=['GET', 'POST'])
def analysis():
    if request.method=='POST':
        link=request.form['link']
        if is_url_scrapable(link):
            driver = process_link(link)
            extracted_text = extract_text(driver)
            driver.quit()
            csv_data = splitter(extracted_text)
            array_to_csv(csv_data, "output.csv")
            df=clean_csv()
            df.dropna(inplace = True)
            df.drop_duplicates(inplace = True)


            column_name = df.columns[0]

            df[column_name] = df[column_name].apply(remove_punct)
            df[column_name] = df[column_name].apply(tokenize)
            df[column_name] = df[column_name].apply(lemmatize)
            df[column_name] = df[column_name].apply(return_sentences)

            df.dropna(inplace = True)
            df.drop_duplicates(inplace = True)


            # print(df[column_name])

            # print(df)

            tfidf_vectorizer = joblib.load("tfidf_vectorizer.joblib")

            numpy_array = df[column_name].values

            # print(numpy_array)

            tfidf_array = tfidf_vectorizer.transform(df[column_name]).toarray()

            # print(tfidf_array)

            class_mapping = {0: 'Forced Action', 1: 'Misdirection', 2: 'Obstruction', 3: 'Scarcity', 4: 'Sneaking', 5: 'Social Proof', 6: 'Urgency'}

            probabilities = rf.predict_proba(tfidf_array)

            pre_predicted = rf.predict(tfidf_array)

            rows_above_threshold = np.where((pre_predicted != 1) & (np.max(probabilities, axis=1) > 0.75))[0]

            if(rows_above_threshold.size<1):
                return render_template('index.html')

            filtered_data = numpy_array[rows_above_threshold]

            predicted_classes = rf.predict(tfidf_array[rows_above_threshold])

            result = pd.DataFrame({'Filtered_Data': filtered_data, 'Predicted_Class': predicted_classes})
            result.replace({"Predicted_Class": class_mapping}, inplace=True)

            category_frequencies = result['Predicted_Class'].value_counts()

            frequency_df = pd.DataFrame({'Category': category_frequencies.index, 'Frequency': category_frequencies.values})

            freq_data= frequency_df.values
            result_val=result.values
            print(result)

            return render_template('analysis.html',freq_data=freq_data)
    
        else:
            return "Invalid URL. Please provide a valid URL."
    else:
        return render_template('analysis.html')
        
@app.route('/report',methods=['GET','POST'])
def report():
    if request.method=='POST':
        fir_nam= request.form['firstname']
        las_nam= request.form['lastname']
        mob= request.form['phone']
        mail= request.form['email']
        new_user= user(first_name=fir_nam,last_name=las_nam,mobile=mob,email=mail)

        try:
            db.session.add(new_user)
            db.session.commit()
            return redirect('/')
        except:
            return render_template('index.html')


    

if __name__ == '__main__':
    app.run(debug=True)
