import torch
import torch.nn as nn
import torch.optim as optim

def model_train(model, criterion, optimizer, num_epochs, trainloader, device):
    # Train the model
        for epoch in range(num_epochs):
            curr_loss = 0.0

            for i, data in enumerate(trainloader, 0):
                # Get the inputs and labels
                inputs, labels = data

                # Move inputs and labels to GPU
                inputs, labels = inputs.to(device), labels.to(device)

                # Zero out the gradient
                optimizer.zero_grad()

                # Forward pass
                outputs = model(inputs)

                # Calculate the loss
                loss = criterion(outputs, labels)

                # Backward pass
                loss.backward()

                # Optimize
                optimizer.step()

                # Add the loss of the current prediction to the running loss
                curr_loss += loss.item()

                # Print statistics every 100 mini batches
                if (i % 100 == 0 and i != 0):
                    # print the epoch, batch and the avarage loss of each prediction in that batch
                    print('epoch: %d, %d - %d loss: %.5f' %(epoch, i-100, i, curr_loss/100))

                    # Reset the running loss
                    curr_loss = 0.0

    # Save the model
        model_path = 'mnist_cnn_model.pth'
        torch.save(model.state_dict(), model_path)
        print(f'Model saved to {model_path}\n')
