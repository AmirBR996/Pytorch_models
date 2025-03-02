import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('./three.csv')
X = df[['Height_cm','Weight_kg','Annual_Income_USD']].values
Y = df['Class'].values

scaler = StandardScaler()
X_train = scaler.fit_transform(X)

X_tensor = torch.from_numpy(X_train).float()
Y_tensor = torch.from_numpy(Y).float().view(-1, 1)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, X):
        return torch.sigmoid(self.linear(X))

epochs = 1000
learning_rate = 0.001

model = Model()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    y_pred = model(X_tensor)
    loss = criterion(y_pred, Y_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")

def make_prediction(data):
    model.eval()
    with torch.no_grad():
        new_data = scaler.transform(data)
        new_tensor = torch.from_numpy(new_data).float()
        prediction = model(new_tensor)
        predicted_class = (prediction > 0.5).float()
    return predicted_class

data = np.array([[168.72700594236812,94.94221523080014,38562.29639046787],[150,55,2000000.9786],[197.72,94.94,38562.296]])
result = make_prediction(data)
print(result)
