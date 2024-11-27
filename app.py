!pip install joblib  # Install joblib if you haven't already

from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Attempt to fix the model file
try:
    with open('housing_model.pkl', 'rb') as f:
        content = f.read()
    corrected_content = content.replace(b'mport joblib\r', b'import joblib')
    with open('housing_model.pkl', 'wb') as f:
        f.write(corrected_content)
except FileNotFoundError:
    print("housing_model.pkl not found. Please make sure you have trained and saved your model.")
except Exception as e:
    print(f"Error while fixing model file: {e}")

# Load the saved model
try:
    model = joblib.load('housing_model.pkl')
except FileNotFoundError:
    print("housing_model.pkl not found. Please make sure you have trained and saved your model.")
    exit() # Exit the program if model file is not found
except Exception as e:
    print(f"Error loading the model: {e}")
    exit() # Exit the program if there is an error loading the model

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
