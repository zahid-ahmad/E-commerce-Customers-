from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

with open('ecommerce_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    features_array = np.array(features).reshape(1, -1)
    prediction = model.predict(features_array)

    return f"Predicted yearly amount spent: {prediction[0][0]:.2f}"

if __name__ == '__main__':
    app.run(debug=True)
