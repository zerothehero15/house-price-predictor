from flask import Flask, request, render_template
import joblib
import numpy as np
import os

app = Flask(__name__, template_folder='index.html')

# Load the saved model with error handling
try:
    model_path = os.path.join(os.getcwd(), 'housing_model.pkl')
    model = joblib.load(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', prediction_text="Error: Model not loaded!")

    # Get input data from the form
    try:
        size = float(request.form['size'])
        bedrooms = int(request.form['bedrooms'])
        bathrooms = int(request.form['bathrooms'])
        location = int(request.form['location'])  # Assuming location is encoded
        year_built = int(request.form['year_built'])
        lot_size = float(request.form['lot_size'])
    except ValueError as e:
        return render_template('index.html', prediction_text="Error: Invalid input values!")
    
    try:
        # Create input array for the model
        input_features = np.array([[size, bedrooms, bathrooms, location, year_built, lot_size]])

        # Predict the price
        prediction = model.predict(input_features)[0]
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error during prediction: {e}")

    return render_template('index.html', prediction_text=f'Predicted House Price: ${prediction:,.2f}')

if __name__ == "__main__":
    app.run(debug=True)
