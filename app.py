from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

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

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from request
    data = request.json

    # Make prediction
    prediction = model.predict([data['features']])

    # Return prediction
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
