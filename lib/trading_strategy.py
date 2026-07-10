import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Load the historical data
df = pd.read_csv('historical_data.csv')

# Define the model architecture and load the saved model
model = nn.Linear(6, 1)
model.load_state_dict(torch.load("model_all_lags.pth"))
model.eval()

#converting lags to tensor and making prediction
lags = df.iloc[-1, -6:].values.astype(float)
lags_tensor = torch.tensor(lags, dtype=torch.float32)

#the prediction through matrix multiplication of lags and model weights, adding the bias
prediction = model(lags_tensor.unsqueeze(0)).item()
print(f"Prediction for {prediction}")