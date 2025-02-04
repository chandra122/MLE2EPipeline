import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend which doesn't require GUI

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import mysql.connector
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import time
import requests
import logging
import joblib
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'root',
    'database': 'co2_emission_db'
}

# Load and preprocess data
def load_and_preprocess_data():
    data = pd.read_csv(r"CO2_emission.csv")
    data = data.drop(columns=['Model', 'Vehicle_Class'])
    data = pd.get_dummies(data, drop_first=True)
    X = data.drop(columns=['CO2_Emissions'])
    y = data['CO2_Emissions']
    return train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_test, y_train, y_test = load_and_preprocess_data()
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train and save models
models = {
    'linear': LinearRegression(),
    'decision_tree': DecisionTreeRegressor(random_state=42),
    'random_forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    joblib.dump(model, f'{name}_model.joblib')

# Database operations
def get_db_connection():
    return mysql.connector.connect(**DB_CONFIG)

def save_prediction(input_data, prediction, model_type, response_time):
    conn = get_db_connection()
    cursor = conn.cursor()
    query = """
    INSERT INTO predictions (
        make, model_year, engine_size, cylinders, transmission,
        fuel_consumption_city, fuel_consumption_hwy, fuel_consumption_comb,
        smog_level, predicted_co2, model_type, response_time, timestamp
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    values = (
        input_data['Make'], input_data['Model_Year'], input_data['Engine_Size'],
        input_data['Cylinders'], input_data['Transmission'],
        input_data['Fuel_Consumption_in_City(L/100 km)'],
        input_data['Fuel_Consumption_in_City_Hwy(L/100 km)'],
        input_data['Fuel_Consumption_comb(L/100km)'],
        input_data['Smog_Level'], prediction, model_type, response_time,
        datetime.now()
    )
    cursor.execute(query, values)
    conn.commit()
    cursor.close()
    conn.close()

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        app.logger.info("Received prediction request")
        data = request.json.get('data')
        model_type = request.json.get('model', 'linear')
        if data is None:
            raise ValueError("Input data is empty")
        if not data:
            raise ValueError("Input data is empty")
        
        input_df = pd.DataFrame([data])
        input_df = pd.get_dummies(input_df, drop_first=True)
        
        for col in X_train.columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[X_train.columns]
        
        input_scaled = scaler.transform(input_df)

        model = joblib.load(f'{model_type}_model.joblib')
        
        start_time = time.time()
        prediction = model.predict(input_scaled)[0]
        response_time = time.time() - start_time

        save_prediction(data, float(prediction), model_type, response_time)
        return jsonify({'prediction': float(prediction), 'response_time': response_time})
    except ValueError as e:
        app.logger.error(f"Value error occurred: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 400
    except FileNotFoundError as e:
        app.logger.error(f"File not found error occurred: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500
    except mysql.connector.Error as e:
        app.logger.error(f"MySQL error occurred: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        app.logger.error(f"An error occurred: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

# Model performance route
@app.route('/model_performance', methods=['GET'])
def model_performance():
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        results[name] = {
            'MSE': mse,
            'R2': r2,
            'CV_scores': cv_scores.tolist(),
            'CV_mean': cv_scores.mean()
        }
    return jsonify(results)

# Feature importance route
@app.route('/feature_importance', methods=['GET'])
def feature_importance():
    model_type = request.args.get('model', 'random_forest')
    model = joblib.load(f'{model_type}_model.joblib')
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_)
    else:
        return jsonify({'error': 'Model does not support feature importance'}), 400
    
    feature_imp = pd.DataFrame({'feature': X_train.columns, 'importance': importances})
    feature_imp = feature_imp.sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_imp.head(10))
    plt.title(f'Top 10 Feature Importances ({model_type.capitalize()})')
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    
    return send_file(img, mimetype='image/png')

# Generate response time graph
@app.route('/monitor', methods=['POST'])
def monitor():
    try:
        input_data = request.json.get('data')
        if input_data is None:
            raise ValueError("Input data is empty")
        if not input_data:
            raise ValueError("Input data is empty")
        
        response_times = []
        predictions = []
        iterations = 50

        for i in range(iterations):
            try:
                start_time = time.time()
                response = requests.post('http://127.0.0.1:5000/predict', json={'data': input_data, 'model': 'linear'})
                response.raise_for_status()  # Raise an exception for HTTP errors
                end_time = time.time()
                response_times.append(end_time - start_time)
                response_dict = response.json()
                if 'prediction' in response_dict:
                    predictions.append(response_dict['prediction'])
                else:
                    app.logger.error(f"Error in iteration {i+1}: {response_dict}")
            except requests.exceptions.RequestException as e:
                app.logger.error(f"Request error occurred in iteration {i+1}: {str(e)}")
        
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
            
            ax1.plot(range(1, iterations+1), response_times, marker='o')
            ax1.set_title('Response Time vs Iteration')
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Response Time (seconds)')
            ax1.grid(True)
            
            ax2.plot(range(1, iterations+1), predictions, marker='o', color='green')
            ax2.set_title('Prediction vs Iteration')
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Predicted CO2 Emission')
            ax2.grid(True)
            
            plt.tight_layout()
            img = io.BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            plt.close('all')
            
            img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
            return jsonify({
                'graph': img_base64,
                'response_times': response_times,  # Add response_times to the response
                'predictions': predictions,        # Add predictions to the response
                'avg_response_time': np.mean(response_times),
                'avg_prediction': np.mean(predictions) if predictions else None
            })
        except Exception as e:
            app.logger.error(f"An error occurred in plot generation: {str(e)}", exc_info=True)
            return jsonify({'error': str(e)}), 500
    except ValueError as e:
        app.logger.error(f"Value error occurred: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        app.logger.error(f"An error occurred in monitor: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
