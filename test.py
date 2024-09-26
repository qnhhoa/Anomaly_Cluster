
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import json
import os
import pyodbc
from resample_test.py import update_new_start_end_timestamp,log_to_hostname_csv, check_features_before_after_resample, matrixname_row_to_column, hostname_csv_to_feature_dict, get_start_end_timestamp
from datetime import datetime, timedelta

app = Flask(__name__)


@app.route('/export_model', methods=['GET'])
def export_model():
    try:
        # Load the trained model
        model_path = 'trained_iforest_model.pkl'
        if not os.path.exists(model_path):
            return jsonify({"error": "Model file not found. Please train the model first."})
        
        iforest = joblib.load(model_path)
        
        # Convert the model to JSON
        model_json = {
            "estimators_": [estimator.get_params() for estimator in iforest.estimators_],
            "max_samples_": iforest.max_samples_,
            "max_features_": iforest.max_features_,
            "contamination": iforest.contamination,
            "n_estimators": iforest.n_estimators,
            "random_state": iforest.random_state,
            "base_estimator": iforest.base_estimator_.get_params()
        }
        
        # Save the JSON to a file
        json_path = 'trained_iforest_model.json'
        with open(json_path, 'w') as json_file:
            json.dump(model_json, json_file)
        
        return jsonify({"message": "Model exported to JSON successfully", "json_path": json_path})
    
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Load the trained model
        model_path = 'trained_iforest_model.pkl'
        feature_detect = 'unifeatures_model.pkl'

        if not os.path.exists(model_path):
            return jsonify({"error": "Model file not found. Please train the model first."})
        
        iforest = joblib.load(model_path)
        iforest_feature = joblib.load(feature_detect)
        
        # Load the new dataset from the request
        data = request.json['data']
        df = pd.DataFrame(data)
   
        # Preprocess the dataset using functions from resample_test.py
        df = log_to_hostname_csv(df=df, save_to='')
        hostnames = [''] //// truyền vào
        df = hostname_csv_to_feature_dict(hostnames, resample_time='60S')
        get_start_end_timestamp(df,hostnames)
        start_timestamp=(datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")+ " 00:00:01"
        end_timestamp= (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")+ " 23:59:01"
        hostname_df_dict = update_new_start_end_timestamp(df,hostnames,start_timestamp,end_timestamp, save_to_pickle=0, filename='')


        # Preprocess the dataset
        df = df.dropna()  # Drop rows with missing values
        # setting first name as index column
        df.set_index("Timestamp", inplace = True)
        # Scale the data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(df)
        
        # Predict anomalies
        # df['Anomaly_Score'] = iforest.fit_predict(data_scaled)
        predictions = iforest.predict(data_scaled)
        anomalies = df[predictions == -1]
        
        # Train a separate model on each feature of the anomalies
        feature_importances = {}
        for feature in anomalies.columns:
            feature_data = anomalies[[feature]]
            iforest_feature.fit(feature_data)
            feature_importances[feature] = iforest_feature.decision_function(feature_data).mean()
        top_5_contributions = sorted(feature_importances.items(), key=lambda item: item[1], reverse=True)[:5]

        # Save results to SQL Server database
        conn_str = (
            "DRIVER={ODBC Driver 17 for SQL Server};"
            "SERVER=172.30.27.239;"
            "DATABASE=AI_TEST;"
            "UID=BongPPQ;"
            "PWD=Abc@123"
        )
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()

        # Create table if not exists
        create_table_query = """
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='AnomalyResults' AND xtype='U')
        CREATE TABLE AnomalyResults (
            Timestamp DATETIME,
            Anomaly_Score INT,
            Feature NVARCHAR(255)
        )
        """
        cursor.execute(create_table_query)
        conn.commit()

        # Insert anomaly results into the table
        insert_query = "INSERT INTO AnomalyResults (Timestamp, Anomaly_Score, Feature) VALUES (?, ?, ?)"
        for index, row in anomalies.iterrows():
            sorted_features = sorted(feature_importances.items(), key=lambda item: item[1], reverse=True)[:5]
            for feature in sorted_features:
                cursor.execute(insert_query, index, -1, feature)
        conn.commit()

        # Close the connection
        cursor.close()
        conn.close()

        return jsonify({"anomalies": anomalies.index.tolist(), "feature_importances": feature_importances})
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)