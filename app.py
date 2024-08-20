from flask import Flask, render_template, request, redirect, url_for, jsonify, flash
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import nltk
from nltk import WordNetLemmatizer
import string
from joblib import load
import csv
import pandas as pd
import numpy as np
import re
import requests
import plotly.graph_objs as go
from flask_pymongo import PyMongo
from pymongo.errors import PyMongoError
import base64, os
from werkzeug.utils import secure_filename
from flask_cors import CORS
import joblib
from nltk.tokenize import sent_tokenize
from selenium.webdriver.common.by import By
import time
from selenium_stealth import stealth 
# Initialize Flask application
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

# Helper functions
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



app.config['UPLOAD_FOLDER'] = 'uploads'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

try:
    app.config["MONGO_URI"] = "mongodb+srv://sarthak062004:PJzsSQUhprD9JAdX@pattern.wkvxu9i.mongodb.net/test?retryWrites=true&w=majority&appName=Pattern"
    mongo = PyMongo(app)
    db = mongo.db
    patterns = db.patterns
    userin = db.userin
    print("connected")
except PyMongoError as e:
    print(f"MongoDB error: {e}")

# Define your routes
@app.route('/receive_url', methods=['POST'])
def receive_url():
    data = request.json
    url = data.get('url')
    # Process URL as needed
    return jsonify({"status": "received", "url": url})

def process_link(link):
    options = Options()
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--headless')
    # options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_argument("--disable-3d-apis")
    options.add_argument('--disable-popup-blocking')
    options.add_argument('--start-maximized')
    options.add_argument('--disable-extensions')
    options.add_argument('--no-sandbox')
    options.add_argument('--ignore-certificate-errors')
    options.add_argument('--allow-running-insecure-content')
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    driver = webdriver.Chrome(options=options)
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    
    stealth(driver,
        user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.53 Safari/537.36',
        languages=["en-US", "en"],
        locale="en-US",
        vendor="Google Inc.",
        platform="Win32",
        webgl_vendor="Google Inc.",
        renderer="Intel Iris OpenGL Engine",
        fix_hairline=True)
    driver.get(link)
    return driver

def is_url_scrapable(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def extract_text(driver):
    i = 0
    extracted_text = ''
    while driver.execute_script("return document.readyState") != "complete":
        pass 
    i = 0
    while (i < 5):
        extracted_text += driver.find_element(By.XPATH, "/html/body").text + '\n'
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
    df = pd.read_csv("uploads/output.csv")
    column_name = df.columns[0]
    df = df[df[column_name].apply(has_letters)]
    df.reset_index(drop=True, inplace=True)
    return df

def array_to_csv(csv_data, filename):
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        for row in csv_data:
            csv_writer.writerow([row])

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

app.secret_key = os.getenv('FLASK_SECRET_KEY', 'default_secret_key')
@app.route('/data_collect', methods=['POST'])
def data_collect():
    if request.method == 'POST':
        fname = request.form.get('fname')
        email = request.form.get('mail')
        url = request.form.get('url')
        dark_pattern_noticed = request.form.get('dark_pattern_noticed')
        file = request.files.get('file')
        encoded_string = None
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            with open(file_path, 'rb') as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            os.remove(file_path)
        userin.insert_one({
            'name': fname,
            'email': email,
            'URL': url,
            'note': dark_pattern_noticed,
            'image_base64': encoded_string
        })
        flash('Form submitted successfully!')
        return redirect(url_for('index'))

@app.route('/analysis', methods=['POST', 'GET'])
def analysis():
    if (request.method == 'POST'):
        input_data = request.json
        link = input_data.get('link')
        driver = process_link(link)
        extracted_text = extract_text(driver)
        driver.quit()
        csv_data = splitter(extracted_text)
        # print(csv_data)
        # array_to_csv(csv_data, "uploads/output.csv")
        # Load your machine learning model
        rf = joblib.load("voting_rf_lr_svc.joblib")
        df = pd.DataFrame({'col':csv_data})
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)
        column_name = df.columns[0]
        df[column_name] = df[column_name].astype(str)
        df[column_name] = df[column_name].apply(lambda x: sent_tokenize(x) if isinstance(x, str) else [])
        df = df.explode(column_name).reset_index(drop=True)
        df[column_name] = df[column_name].apply(remove_punct)
        df[column_name] = df[column_name].apply(tokenize)
        df[column_name] = df[column_name].apply(lemmatize)
        df[column_name] = df[column_name].apply(lambda x: None if (len(x) > 10 or len(x) < 2) else x)
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)
        df[column_name] = df[column_name].apply(return_sentences)
        if df.empty:
            response_json = {
                    "id": 2
                }
            return jsonify(response_json)
        class_mapping = {
            0: 'forced action', 1: 'misdirection', 2: 'obstruction', 
            3: 'scarcity', 4: 'sneaking', 5: 'social proof', 6: 'urgency'
        }

        tfidf_vectorizer = joblib.load("vectorizer2.joblib")
        numpy_array = df[column_name].values
        tfidf_array = tfidf_vectorizer.transform(df[column_name])
        
        probabilities = rf.predict_proba(tfidf_array)
        # rows_above_threshold = np.where(np.max(probabilities, axis=1) > 0.80)[0]
        rows_above_threshold = []
        thresholds = {
            0: 0.60,  # Threshold for 'forced action'
            1: 0.80,  # Threshold for 'misdirection'
            2: 0.60,  # Threshold for 'obstruction'
            3: 0.85,  # Threshold for 'scarcity'
            4: 0.50,  # Threshold for 'sneaking'
            5: 0.85,  # Threshold for 'social proof'
            6: 0.75   # Threshold for 'urgency'
        }
        # Iterate over each row in the probabilities array
        for i, probs in enumerate(probabilities):
            # Check if the probability for the predicted class is above its specific threshold
            predicted_class = np.argmax(probs)
            if probs[predicted_class] > thresholds[predicted_class]:
                rows_above_threshold.append(i)

        rows_above_threshold = np.array(rows_above_threshold)
        # os.remove("uploads/output.csv")
        if rows_above_threshold.size < 1:
            response_json = {
                "id": 0
            }
            return jsonify(response_json)
        filtered_data = numpy_array[rows_above_threshold]
        predicted_classes = rf.predict(tfidf_array[rows_above_threshold])
        confidence_scores = np.max(probabilities[rows_above_threshold], axis=1)
        result = pd.DataFrame({'Filtered_Data': filtered_data, 'Predicted_Class': predicted_classes, 'Confidence_Score': confidence_scores})
        p = {}
        for i in range(0, len(filtered_data)):
            # print(predicted_classes[i], filtered_data[i])
            if (class_mapping[predicted_classes[i]] in p):
                p[class_mapping[predicted_classes[i]]].append(filtered_data[i])
            else:
                p[class_mapping[predicted_classes[i]]] = [filtered_data[i]]

        result.replace({"Predicted_Class": class_mapping}, inplace=True)
        category_frequencies = result['Predicted_Class'].value_counts()
        frequency_df = pd.DataFrame({'Category': category_frequencies.index, 'Frequency': category_frequencies.values})
        numpy_data = frequency_df.to_numpy()
        column1 = numpy_data[:, 0]
        column2 = numpy_data[:, 1]
        for key, values in p.items():
            patterns.update_one(
                {'_id': key},  
                {'$addToSet' : {'data': {'$each': values}}}, 
                upsert = True  
                    # Create the document if it does not exist
            )
        trace = go.Bar(x=column1, y=column2, name='Data')
        layout = go.Layout(
            title='Dark Pattern Found', 
            xaxis=dict(title='Dark Pattern Category'), 
            yaxis=dict(title='Occurrences'),
            autosize=False,
            margin=dict(l=50, r=50, b=100, t=100)  # Adjust margins as needed
        )
        fig = go.Figure(data=[trace], layout=layout)
        fig_json = fig.to_json()
        response_json = {
            "id": 1,
            "data":fig_json
        }
        response = jsonify(response_json)
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        print(result)
        return response
        # else:
        #     response_json = {
        #         "id": 2
        #     }
        #     return jsonify(response_json)
    else:
        return render_template('analysis.html')

port = int(os.environ.get('PORT', 5000))
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=port, debug=True)
