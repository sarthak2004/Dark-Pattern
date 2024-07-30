from flask import Flask, render_template, request, redirect, url_for,jsonify,flash
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
# from flask_sqlalchemy import SQLAlchemy
import nltk
from nltk import WordNetLemmatizer
# import matplotlvib.pyplot as plt
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
import plotly.graph_objs as go
from flask_pymongo import PyMongo






def remove_punct(text):
    if isinstance(text, str):
        return "".join([ch for ch in text if ch not in string.punctuation])
    else:
        return str(text)


def tokenize(text):
    text = re.split('\s+' ,text)
    return [x.lower() for x in text]



def remove_stopwords(text):
    return [word for word in text if word not in nltk.corpus.stopwords.words('english')]


def lemmatize(text):
    word_net = WordNetLemmatizer()
    return [word_net.lemmatize(word) for word in text]


def return_sentences(tokens):
    return " ".join([word for word in tokens])


rf= joblib.load("Random_forest_final.joblib")




app = Flask(__name__)
app.config["MONGO_URI"] = "mongodb://localhost:27017/myDatabase"

db = PyMongo(app).db




url=""
@app.route('/receive_url', methods=['POST'])
def receive_url():
    data = request.json
    url = data.get('url')

def process_link(link):
    options = Options()
    options.add_argument('--disable-dev-shm-usage')
    # options.add_argument("--headless=new")
    driver = webdriver.Chrome(service=Service(r'C:\Users\mansi\.wdm\drivers\chromedriver\win64\127.0.6533.72\chromedriver-win32\chromedriver.exe'), options=options)

   
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


@app.route('/form-submit',methods=['GET','POST'])
def form_submit():
    if request.method=='POST':
        flash(f"Dark Pattern reported successfully")
    return render_template("index.html")
    
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


            class_mapping= {0: 'forced action', 1: 'misdirection', 2: 'obstruction', 3: 'scarcity', 4: 'sneaking', 5: 'social proof', 6: 'urgency'}

            tfidf_vectorizer = joblib.load("vectorizer2.joblib")

            numpy_array = df[column_name].values


            tfidf_array = tfidf_vectorizer.transform(df[column_name])


            probabilities = rf.predict_proba(tfidf_array)

            pre_predicted = rf.predict(tfidf_array)

            rows_above_threshold = np.where( (np.max(probabilities, axis=1) > 0.80))[0]

            if(rows_above_threshold.size<1):
                return render_template('error.html',link=link)

            filtered_data = numpy_array[rows_above_threshold]

            predicted_classes = rf.predict(tfidf_array[rows_above_threshold])

            result = pd.DataFrame({'Filtered_Data': filtered_data, 'Predicted_Class': predicted_classes})
            result.replace({"Predicted_Class": class_mapping}, inplace=True)

            category_frequencies = result['Predicted_Class'].value_counts()

            frequency_df = pd.DataFrame({'Category': category_frequencies.index, 'Frequency': category_frequencies.values})


            numpy_data= (frequency_df).to_numpy()

            result_val=result.values

            numpy_data= (frequency_df).to_numpy()
            result_val=result.values
            print(result)

            column1 = numpy_data[:, 0]
            column2 = numpy_data[:, 1]

            # Creating plotly graph
            trace = go.Bar(x=column1, y=column2, name='Data')
            layout = go.Layout(
                title='Dark Pattern Found', 
                xaxis=dict(title='Dark Pattern Category'), 
                yaxis=dict(title='Occurences'),
                autosize=False,
                width=800,  # Adjust width as needed
                height=600,  # Adjust height as needed
                margin=dict(l=50, r=50, b=100, t=100)# Adjust margins as needed
            )
            fig = go.Figure(data=[trace], layout=layout)
            graph = fig.to_html(full_html=False, default_height=500, default_width=700)

            return render_template('analysis.html',graph=graph)
    
        else:
            return render_template('not_scrappable.html',link=link)
    else:
        return render_template('analysis.html')

    
    

if __name__ == '__main__':
    app.run(debug=True)