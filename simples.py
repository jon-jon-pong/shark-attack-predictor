import torch
import torch.nn as nn
import torch.optim as optim

# Create a simple neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 5)  # Input layer (10 features) to hidden layer (5 neurons)
        self.fc2 = nn.Linear(5, 1)   # Hidden layer to output layer (1 output)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create some dummy data
X = torch.randn(100, 10)  # 100 samples, 10 features each
y = torch.randn(100, 1)   # 100 target values

# Initialize the model, loss function, and optimizer
model = SimpleNet()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(100):
    # Forward pass
    predictions = model(X)
    loss = criterion(predictions, y)
    
    # Backward pass and optimization
    optimizer.zero_grad()  # Clear gradients
    loss.backward()        # Compute gradients
    optimizer.step()       # Update weights
    
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')

# Make a prediction
with torch.no_grad():
    test_input = torch.randn(1, 10)
    prediction = model(test_input)
    print(f'\nTest prediction: {prediction.item():.4f}')