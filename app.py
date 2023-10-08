from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression


app = Flask(__name__)

data = pd.read_csv('helium_balloon_dataset.csv')

X = data[['Number of Guests', 'Type of Event']]
y = data['Number of Helium Balloons']

# Encode categorical variables (Type of Event) using one-hot encoding
X_encoded = pd.get_dummies(X, columns=['Type of Event'], drop_first=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# One-hot encoding mapping for event types
event_mapping = {
    'Birthday': [1, 0, 0, 0],
    'Corporate Event': [0, 1, 0, 0],
    'Graduation': [0, 0, 1, 0],
    'Holiday': [0, 0, 0, 1]
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    num_guests = int(request.form['num_guests'])
    event_type = request.form['event_type']

    # Encode the event type using the mapping
    event_encoded = event_mapping.get(event_type, [0, 0, 0, 0])

    # Create a DataFrame with user input
    new_data = pd.DataFrame({
        'Number of Guests': [num_guests],
        'Type of Event_Birthday': event_encoded[0],
        'Type of Event_Corporate Event': event_encoded[1],
        'Type of Event_Graduation': event_encoded[2],
        'Type of Event_Holiday': event_encoded[3]
    })

    # Make a prediction
    predicted_balloons = model.predict(new_data)

    return render_template('result.html', predicted_balloons=int(predicted_balloons[0]))

if __name__ == '__main__':
    app.run(debug=True)
