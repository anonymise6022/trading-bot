import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Load the data (depending on user's choice)
user_input = input("Which results do you want to use? 40k or latest?: ")

if user_input == "latest":
    df = pd.read_csv('historical_data.csv')
else:
    df = pd.read_csv('BTCUSDT-1h.csv')

use_oi = True  # Set to True if you want to use open interest data, False otherwise

checkpoint = torch.load("model_price_oi.pth" if use_oi else "model_price_only.pth")
feature_cols = checkpoint['features']
model_size = checkpoint['model_size']

# Define the model architecture and load the saved model
model = nn.Linear(model_size, 1)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

#converting lags to tensor and making prediction
latest_features = df[feature_cols].iloc[-1].values.astype(float)
features_tensor = torch.tensor(latest_features, dtype=torch.float32)

#the prediction through matrix multiplication of lags and model weights, adding the bias
prediction = model(features_tensor.unsqueeze(0)).item()
print(f"Prediction for {prediction}")