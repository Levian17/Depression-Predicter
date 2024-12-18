from src.model_definition import classify_new_data as predict
from src.model_definition import train_model
import joblib

train_model()

encoder = joblib.load('encoder.joblib')
scaler = joblib.load('scaler.joblib')
new_data = ['Female', 23.0, 10.0, 3.0, 3.0, 3.0, 'More than 8 hours', 'Healthy', 'No', 'Yes']
model_path = 'depression_model.pth'
predicted_class = predict(new_data, encoder, scaler, model_path)
print(f"Predicted Class: {predicted_class}")