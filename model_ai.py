import numpy as np
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    torch_available = True
except ImportError:
    print("Torch is not installed. Please install it using 'pip install torch'")
    torch_available = False

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

if torch_available:
    # Check if CUDA (GPU) is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. LINEAR REGRESSION
    # Generate dummy data
    X = np.random.rand(100, 1) * 10  # Fitur
    y = 3 * X.squeeze() + np.random.randn(100) * 2  # Target

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model_lr = LinearRegression()
    model_lr.fit(X_train, y_train)
    y_pred = model_lr.predict(X_test)

    print("Linear Regression MSE:", mean_squared_error(y_test, y_pred))

    # 2. TWO-LAYER NEURAL NETWORK
    class TwoLayerNN(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(TwoLayerNN, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, output_size)
        
        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x

    # Dummy data for training
    X_nn = torch.rand(100, 1).to(device)
    y_nn = (3 * X_nn + torch.randn(100, 1) * 0.5).to(device)

    # Define model, loss, and optimizer
    model_nn = TwoLayerNN(1, 10, 1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model_nn.parameters(), lr=0.01)

    # Training loop
    for epoch in range(500):
        optimizer.zero_grad()
        output = model_nn(X_nn)
        loss = criterion(output, y_nn)
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

    # 3. OPTIMIZATION COMPARISON
    optim_sgd = optim.SGD(model_nn.parameters(), lr=0.01)
    optim_adam = optim.Adam(model_nn.parameters(), lr=0.01)
    print("Comparison of optimizers can be run by changing optimizer variable")

    print("AI Model implementation completed successfully.")


