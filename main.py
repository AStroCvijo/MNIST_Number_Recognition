import torch
import torch.nn as nn
import torch.optim as optim

from utils.argparser import arg_parse

from model.model import MNIST_CNN_Model

from data.data_download_and_load import load_data

from train.trainer import model_train
from train.evaluation import model_eval

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Parse arguments
    args = arg_parse()

    # Create the model and train it with the given parameters
    if args.train:
        # Load the dataset
        trainloader, testloader = load_data()

        # Initialize and move the model to the GPU
        model = MNIST_CNN_Model()
        model.to(device)

        # Initialize training parameters
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr = args.learning_rate)
        num_epochs = args.epochs

        # Train the model
        model_train(model, criterion, optimizer, num_epochs, trainloader, device)

        # Evaluation of the model
        model_eval(model, testloader, device)