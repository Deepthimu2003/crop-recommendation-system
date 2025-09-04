from flask import Flask, request, render_template
import numpy as np
import pandas
import sklearn
import pickle

# Load the trained model and scalers
model = pickle.load(open('model.pkl', 'rb'))
sc = pickle.load(open('standscaler.pkl', 'rb'))
mx = pickle.load(open('minmaxscaler.pkl', 'rb'))

# Create Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    # Get input values from form
    N = float(request.form['Nitrogen'])
    P = float(request.form['Phosporus'])
    K = float(request.form['Potassium'])
    temp = float(request.form['Temperature'])
    humidity = float(request.form['Humidity'])
    ph = float(request.form['pH'])
    rainfall = float(request.form['Rainfall'])

    # Preprocess input
    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)
    mx_features = mx.transform(single_pred)
    sc_mx_features = sc.transform(mx_features)

    # Predict probabilities
    probabilities = model.predict_proba(sc_mx_features)[0]
    top_indices = np.argsort(probabilities)[::-1][:6]  # top 6: best + 5 alternatives

    # Crop class index to name mapping
    crop_dict = {
        1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
        8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
        14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
        19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
    }

    # Convert class indices to crop names
    top_crops = [crop_dict.get(index + 1, "Unknown") for index in top_indices]
    best_crop = top_crops[0]
    alternative_crops = top_crops[1:]

    result = f"{best_crop} is the best crop to be cultivated right there."
    alternatives = "Other suitable crops: " + ", ".join(alternative_crops)

    return render_template('index.html', result=result, alternatives=alternatives)

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
