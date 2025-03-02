import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('./two.csv')
X = df[['Feature1', 'Feature2', 'Feature3']].values
Y = df['Class'].values

# scaler = StandardScaler()
# X_train = scaler.fit_transform(X)

X_tensor = torch.from_numpy(X).float()
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

data = np.array([[0.49,0.35,-1.59],[-0.79,-1.35,1.23]])
result = print(make_prediction(data))
print(result)
