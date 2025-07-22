from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import joblib
import logging
from tensorflow.keras.losses import MeanSquaredError


app = Flask(__name__)

# Load model and preprocessors
model = tf.keras.models.load_model(
    "salary_prediction_model.h5",
    custom_objects={'mse': MeanSquaredError()}
)
scaler_x = joblib.load(open('scaler_x.pkl', 'rb'))
scaler_y = joblib.load(open('scaler_y.pkl', 'rb'))
encoders = joblib.load(open('encoders.pkl', 'rb'))
order_columns = joblib.load(open('order_columns.pkl', 'rb'))

# Load metadata
categorical_cols = joblib.load(open('categorical_cols.pkl', 'rb'))
numerical_cols = joblib.load(open('numerical_cols.pkl', 'rb'))
column_values = joblib.load(open('column_values.pkl', 'rb'))

@app.route('/')
def home():
    return render_template(
        'form.html',
        categorical_cols=categorical_cols,
        numerical_cols=numerical_cols,
        column_values=column_values,
        prediction=None
    )

@app.route('/predict', methods=['POST'])
def predict():
    # Collect form data
    input_data = {}
    for col in categorical_cols:
        input_data[col] = request.form[col]
    for col in numerical_cols:
        input_data[col] = float(request.form[col])

    # Convert to DataFrame
    df = pd.DataFrame([input_data])

    # Label encode categorical columns
    for col in categorical_cols:
        encoder = encoders[col]
        df[col] = encoder.transform(df[col])


    # Scale numerical columns
    #df[numerical_cols] = scaler_x.transform(df[numerical_cols])
    
    #model_input_columns = numerical_cols + categorical_cols
        
    #df = df[model_input_columns]

    # Predict salary
    #prediction_scaled = model.predict(df)[0][0]
    
    #prediction = scaler_y.inverse_transform([[prediction_scaled]])[0][0]
    
    df[order_columns] = scaler_x.transform(df[order_columns])
    
    ##df = df[order_columns]

    prediction_scaled = model.predict(df.to_numpy())[0][0]
    prediction = scaler_y.inverse_transform([[prediction_scaled]])[0][0]

    #return f"<h2>Predicted Salary: ₹{prediction:.2f}</h2>"
    
    return render_template(
        'form.html',
        categorical_cols=categorical_cols,
        numerical_cols=numerical_cols,
        column_values=column_values,
        prediction=f"Predicted Salary: ₹{prediction:.2f}"
    )

if __name__ == '__main__':
    app.run(debug=True)
