from flask import Flask, request, jsonify, send_from_directory
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import pandas as pd
import os

app = Flask(__name__)

# Load the dataset
iris_data = pd.read_csv("iris.csv")

# Split features and target variable
X = iris_data.drop(columns=['variety'])  # Features
y = iris_data['variety']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Save the trained model to a file
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

@app.route('/')
def index():
    # Serve the index.html file
    return send_from_directory(os.path.join(app.root_path, ''), '/templates/index.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    # Get data from request
    data = request.json

    # Make prediction
    prediction = model.predict([list(data.values())])

    # Return prediction
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
