from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os
import pickle
# Initialize Flask App
app = Flask(__name__)

file_path = os.path.join(os.getcwd(),"models", 'features.pkl')
file_path_pln= os.path.join(os.getcwd(),"models", 'pipeline.pkl')

# Load the features
with open(file_path, 'rb') as file:
    features = pickle.load(file)

print("Features loaded successfully!")
print("Features:", features)


# Load the features
with open(file_path_pln, 'rb') as file:
    pipeline = pickle.load(file)

print("Features loaded successfully!")
print("Features:", features)



@app.route('/')
def home():
    return render_template('form.html', columns=features)

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    input_data = {col: request.form[col] for col in features}
    input_df = pd.DataFrame([input_data])
    prediction = pipeline.predict(input_df)[0]
    print(prediction)

    label_mapping = {
        0: "Bad",
        1: "Good",
        2: "Not Bad",
        3: "Very Bad",
        4: "Very Good"
    }
# Get the corresponding label
    result = label_mapping.get(prediction, "Unknown")
    return f"The model prediction is: {result}"

if __name__ == '__main__':
    app.run(debug=True,port=5001)