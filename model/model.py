import torch
import torch.nn as nn
import torch.optim as optim

class MNIST_CNN_Model(nn.Module):
    
    # Constructor method
    def __init__(self):
        super(MNIST_CNN_Model, self).__init__()
        
        # First conv layer
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # Second conv layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # First fully connected layer
        self.fc1 = nn.Linear(64*7*7, 128)
        # Second fully connected layer
        self.fc2 = nn.Linear(128, 10)

    # Forward method
    def forward(self, x):
        # Apply first conv layer and ReLU
        x = torch.relu(self.conv1(x))
        # Apply max pooling
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        # Apply second conv layer and ReLU
        x = torch.relu(self.conv2(x))
        # Apply max pooling
        x = torch.max_pool2d(x, kernel_size=2, stride=2)

        # Flatten the output
        x = x.view(-1, 64*7*7)
        # Apply first fully connected layer and ReLU
        x = torch.relu(self.fc1(x))
        # Apply second fully connected layer
        x = self.fc2(x)

        return x