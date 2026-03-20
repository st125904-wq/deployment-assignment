from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load the new fraud model
model = pickle.load(open('fraud_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html') 

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    # Convert the incoming JSON into a Pandas DataFrame for the model pipeline
    df = pd.DataFrame([data])
    
    # Predict
    prediction = model.predict(df)
    
    # Format the result nicely
    result_text = "Fraudulent Transaction" if int(prediction[0]) == 1 else "Legitimate Transaction"
    
    return jsonify({'prediction': result_text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)