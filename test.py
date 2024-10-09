
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import json
import os
import pyodbc
from datetime import datetime, timedelta
# Import necessary functions from resample_test.py
from resample_test import update_new_start_end_timestamp, hostname_csv_to_feature_dict, get_start_end_timestamp, log_to_hostname_csv



app = Flask(__name__)


@app.route('/export_model', methods=['GET'])
def export_model():
    try:
        # Load the trained model
        # model_path = r'C:/Users/abc/Desktop/Zabbix/myproject/trained_iforest_model.pkl/trained_iforest_model.pkl'
        model_path = os.path.join(os.path.dirname(__file__), 'trained_iforest_model.pkl')
        # feature_detect_path = os.path.join(os.path.dirname(__file__), 'unifeatures_model.pkl')
        if not os.path.exists(model_path):
            return jsonify({"error": "Model file not found. Please train the model first."})
        
        iforest = joblib.load(model_path)
        
        # Convert the model to JSON
        model_json = {
            "estimators_": [estimator.get_params() for estimator in iforest.estimators_],
            "max_samples_": iforest.max_samples_,
            # "max_features_": iforest.max_features_,
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

# @app.route('/test', methods=['GET'])
# def predict():
#     try:
#         # Load the trained model
#         model_path = 'trained_iforest_model.pkl'
#         feature_detect = 'unifeatures_model.pkl'
#         conn_str = (
#             "Trusted_Connection:=yes;"
#             "DRIVER={SQL Server};"
#             "SERVER=localhost;"
#             "DATABASE=Test_API;"
#             "UID=sa;"  # Replace with your username
#             "PWD=sa"
#         )
#         conn = pyodbc.connect(conn_str)
#         cursor = conn.cursor()

#         # Retrieve data from the Test table
#         query = "SELECT hostid, hostname, MatrixID, MatrixName, delay, value, Timestamp FROM Test"
#         df = pd.read_sql(query, conn)
#         cursor.close()
#         conn.close()

#         # Convert the DataFrame to a JSON response
#         result = df.to_dict(orient='records')
#         return jsonify({"data": result})
    
#     except Exception as e:
#         return jsonify({"error": str(e)})


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Load the trained model
        model_path = os.path.join(os.path.dirname(__file__), 'trained_iforest_model.pkl')
        feature_detect = os.path.join(os.path.dirname(__file__), 'unifeatures_model.pkl')

        conn_str = (
            "Trusted_Connection:=yes;"
            "DRIVER={SQL Server};"
            "SERVER=localhost;"
            "DATABASE=Test_API;"
            "UID=sa;"  # Replace with your username
            "PWD=sa"
        )
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()

        # Retrieve data from the Test table
        query = "SELECT hostid, hostname, MatrixID, MatrixName, delay, value, Timestamp FROM Test"
        df = pd.read_sql(query, conn)
        df = pd.DataFrame(df)

        if not os.path.exists(model_path):
            return jsonify({"error": "Model file not found. Please train the model first."})
        
        iforest = joblib.load(model_path)
        iforest_feature = joblib.load(feature_detect)
        
        # Load the new dataset from the request
        # data = request.json['data']
        # df = pd.DataFrame(data)
   
        # Preprocess the dataset using functions from resample_test.py
        # df = log_to_hostname_csv(df=df, save_to='')
        hostnames = df['hostname'].unique()  #truyền vào
        # hostnames = ['PRDB-CEN05','PRDB-CEN06']
        # print(hostnames)
        hostname_data = log_to_hostname_csv(df)
        df_transformed = hostname_csv_to_feature_dict(hostname_data,hostnames, resample_time='60S')
        start_timestamp, end_timestamp = get_start_end_timestamp(df_transformed,hostnames)
        print(start_timestamp,  end_timestamp)
        # print("df",df)
        # print(df)
        # start_timestamp=(datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")+ " 00:00:01"
        # end_timestamp= (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")+ " 23:59:01"
        hostname_df_dict = update_new_start_end_timestamp(df_transformed,hostnames,start_timestamp,end_timestamp, save_to_pickle=0, filename='')
        # Flatten the dictionary of DataFrames into a single DataFrame
        # df_list = []
        # for hostname, features in hostname_df_dict.items():
        #     for feature_name, feature_df in features.items():
        #         feature_df['hostname'] = hostname
        #         feature_df['feature'] = feature_name
        #         df_list.append(feature_df)
        # dff = pd.concat(df_list)
        # Preprocess the dataset
        for hostname in hostnames:
            result = hostname_df_dict[hostname]
            print(result)
            result = result.dropna()  # Drop rows with missing values
        # setting first name as index column
            # result.set_index("Timestamp", inplace = True)
        # Scale the data
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(result)
        
        # Predict anomalies
        # df['Anomaly_Score'] = iforest.fit_predict(data_scaled)
            predictions = iforest.predict(data_scaled)
            anomalies = result[predictions == -1]
        
            # Train a separate model on each feature of the anomalies
            feature_importances = {}
            for feature in anomalies.columns:
                feature_data = anomalies[[feature]]
                iforest_feature.fit(feature_data)
                feature_importances[feature] = iforest_feature.decision_function(feature_data).mean()
            top_5_contributions = sorted(feature_importances.items(), key=lambda item: item[1], reverse=True)[:5]


        # Create table if not exists
        create_table_query = """
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='AnomalyResults' AND xtype='U')
        CREATE TABLE AnomalyResults (
            hostname NVARCHAR(255),
            Timestamp DATETIME,
            Anomaly_Score INT,
            Feature NVARCHAR(255)
        )
        """
        cursor.execute(create_table_query)
        conn.commit()

        # Insert anomaly results into the table
        insert_query = "INSERT INTO AnomalyResults (hostname, Timestamp, Anomaly_Score, Feature) VALUES (?, ?, ?, ?)"
        for index, row in anomalies.iterrows():
            sorted_features = sorted(feature_importances.items(), key=lambda item: item[1], reverse=True)[:5]
            for feature in sorted_features:
                cursor.execute(insert_query, index, -1, feature, row['hostname'])
        conn.commit()

        # Close the connection
        cursor.close()
        conn.close()

        return jsonify({"anomalies": anomalies.index.tolist(), "feature_importances": feature_importances})
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)