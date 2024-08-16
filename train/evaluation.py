import torch
import torch.nn as nn
import torch.optim as optim

def model_eval(model, testloader, device):

    # Set the model to evaluation mode
    model.eval()

    # Initialize counters
    correct = 0
    total = 0

    # Disable the gradient and evaluate the model
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print('Accuracy of the network on the test images: %.2f %%' % accuracy)