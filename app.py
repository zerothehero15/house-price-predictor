from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the saved model
model = joblib.load('housing_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the form
    size = float(request.form['size'])
    bedrooms = int(request.form['bedrooms'])
    bathrooms = int(request.form['bathrooms'])
    location = int(request.form['location'])  # Assuming location is encoded
    year_built = int(request.form['year_built'])
    lot_size = float(request.form['lot_size'])
    
    # Create input array for the model
    input_features = np.array([[size, bedrooms, bathrooms, location, year_built, lot_size]])
    
    # Predict the price
    prediction = model.predict(input_features)[0]
    
    return render_template('index.html', prediction_text=f'Predicted House Price: ${prediction:,.2f}')

if __name__ == "__main__":
    app.run(debug=True)
