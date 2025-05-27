import os

model_path = 'bilstm_fraud_detection.keras'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")
