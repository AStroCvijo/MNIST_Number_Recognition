import torch
import torchvision
import torchvision.transforms as transforms

def load_data():
    # Define the transformation
    transform = transforms.Compose([
    transforms.ToTensor(),                # Convert images to PyTorch tensors (scale tha values to a range between 0 and 1)
    transforms.Normalize((0.5,), (0.5,))  # Normalize tensors to have a mean of 0 and a range between -1 and 1
    ])

    # Download and load the MNIST dataset for training
    trainset = torchvision.datasets.MNIST(
        root='./data',          # Where the data will be saved
        train=True,             # Get the training set
        download=True,          # Download the data if not present
        transform=transform     # Apply the transformations
    )

    # Download and load the MNIST dataset for testing
    testset = torchvision.datasets.MNIST(
    root='./data',          # Directory where the data will be saved
    train=False,            # Get the test set
    download=True,          # Download the data if not present
    transform=transform     # Apply the transformations
    )

    # Create a data loader for the training set
    trainloader = torch.utils.data.DataLoader(
        trainset,               # Dataset
        batch_size=64,          # Load 64 samples per batch
        shuffle=True            # Shuffle the data
    )

    # Create a data loader for the test set
    testloader = torch.utils.data.DataLoader(
        testset,                # Dataset
        batch_size=64,          # Load 64 samples per batch
        shuffle=False           # Don't schuffle the data
    )

    return trainloader, testloader