import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork(nn.Module):
    def __init__(self, c):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(2, c)
        self.fc2 = nn.Linear(c, c)
        self.fc3 = nn.Linear(c, 1)

    def forward(self, x):
        x = torch.erf(self.fc1(x))
        x = torch.erf(self.fc2(x))
        x = self.fc3(x)
        return x

def training(train_loader, test_loader, width, learning_rate):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  criterion = nn.MSELoss()
  nb_initialisations = 3
  for i in range(nb_initialisations):
    torch.manual_seed(42 + i)
    model = NeuralNetwork(width).to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    train_losses = []
    test_losses = []
    min_test_loss = []

    epochs = 1000

    for epoch in range(epochs): 
        train_loss = train(model, criterion, optimizer, train_loader, device)
        test_loss = test(model, criterion, test_loader, device)
        if epoch % 50 == 0:
          print(f"Epoch {epoch+1}, Train Loss: {train_loss}, Test Loss: {test_loss}")
        train_losses.append(train_loss)
        test_losses.append(test_loss)

    minimum = np.min(test_losses)
    min_test_loss.append(minimum)

    print(f"Test error : {minimum}, number of samples : {len(train_loader)*16}, initialization : {i+1}")

  return np.mean(min_test_loss)


def train(model, criterion, optimizer, train_loader, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / (len(train_loader))


def test(model, criterion, test_loader, device):
    model.eval()
    running_loss = 0.0
    test_loss = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
    test_loss = running_loss / (len(test_loader))
    return test_loss