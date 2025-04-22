from flask import Flask, render_template, request
import pickle
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Corrected Flask initialization
app = Flask(__name__)

# Function to predict based on input values
def predict(values, dic):
    if len(values) == 8:
        model = pickle.load(open('models/kidney.pkl', 'rb'))
    elif len(values) == 26:
        model = pickle.load(open('models/breast_cancer.pkl', 'rb'))
    elif len(values) == 13:
        model = pickle.load(open('models/heart.pkl', 'rb'))
    elif len(values) == 18:
        model = pickle.load(open('models/kidney.pkl', 'rb'))
    elif len(values) == 10:
        model = pickle.load(open('models/liver.pkl', 'rb'))
    else:
        return "Invalid Input Length"

    values = np.asarray(values)
    return model.predict(values.reshape(1, -1))[0]

# Routes
@app.route("/")
def home():
    return render_template('home.html')

@app.route("/diabetes", methods=['GET', 'POST'])
def diabetesPage():
    return render_template('diabetes.html')

@app.route("/cancer", methods=['GET', 'POST'])
def cancerPage():
    return render_template('breast_cancer.html')

@app.route("/heart", methods=['GET', 'POST'])
def heartPage():
    return render_template('heart.html')

@app.route("/kidney", methods=['GET', 'POST'])
def kidneyPage():
    return render_template('kidney.html')

@app.route("/liver", methods=['GET', 'POST'])
def liverPage():
    return render_template('liver.html')

@app.route("/malaria", methods=['GET', 'POST'])
def malariaPage():
    return render_template('malaria.html')

@app.route("/pneumonia", methods=['GET', 'POST'])
def pneumoniaPage():
    return render_template('pneumonia.html')

@app.route("/predict", methods=['POST', 'GET'])
def predictPage():
    if request.method == 'POST':  
        try:
            to_predict_dict = request.form.to_dict()
            to_predict_list = [float(value) for value in to_predict_dict.values() if value.replace('.', '', 1).isdigit()]
            
            if to_predict_list:
                pred = predict(to_predict_list, to_predict_dict)
            else:
                pred = "Error: No valid numeric input provided."

            return render_template('predict.html', pred=pred)

        except Exception as e:
            print("Error occurred:", e)
            return render_template("home.html", message="Please enter valid Data")

    return render_template("home.html")

@app.route("/malariapredict", methods=['POST', 'GET'])
def malariapredictPage():
    if request.method == 'POST':
        try:
            if 'image' in request.files:
                img = Image.open(request.files['image'])
                img = img.resize((36, 36))
                img = np.asarray(img).reshape((1, 36, 36, 3)).astype(np.float64)

                model = load_model("models/malaria.h5")
                pred = np.argmax(model.predict(img)[0])
                return render_template('malaria_predict.html', pred=pred)

        except:
            return render_template('malaria.html', message="Please upload an Image")

    return render_template('malaria.html')

@app.route("/pneumoniapredict", methods=['POST', 'GET'])
def pneumoniapredictPage():
    if request.method == 'POST':
        try:
            if 'image' in request.files:
                img = Image.open(request.files['image']).convert('L')
                img = img.resize((36, 36))
                img = np.asarray(img).reshape((1, 36, 36, 1)) / 255.0

                model = load_model("models/pneumonia.h5")
                pred = np.argmax(model.predict(img)[0])
                return render_template('pneumonia_predict.html', pred=pred)

        except:
            return render_template('pneumonia.html', message="Please upload an Image")

    return render_template('pneumonia.html')

# Corrected Main Execution Condition
if __name__ == '__main__':
    app.run(debug=True)
