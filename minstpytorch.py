# this is my MNIST project with pytorch.
# importing all the stuff we need.
import torch  # main pytorch library
import torch.nn as nn  # for making neural networks
import torch.optim as optim  # for optimization (like gradient descent)
from torch.utils.data import DataLoader  # helps load data in batches
from torchvision import datasets, transforms  # for getting mnist data
import matplotlib.pyplot as plt  # for plotting graphs
import numpy as np  # for arrays and math stuff
from tqdm import tqdm  # makes cool progress bars

# I check if we have a gpu or just cpu (gpu is way faster!)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# okay so first we need to get the data
print("\n" + "="*50)
print("LOADING MNIST DATASET")
print("="*50)

# this thing transforms our images to tensors and normalizes them
# normalize means making all values between -1 and 1 (helps training)
transform = transforms.Compose([
    transforms.ToTensor(),  # converts image to tensor
    transforms.Normalize((0.1307,), (0.3081,))  # these are mnist mean and std
])

# download and load the training data (60000 images)
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# download and load the test data (10000 images)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# dataloader splits data into batches (64 images at a time for training)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# bigger batches for testing since we dont need gradients
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# lets see how much data we got
print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# function to show some sample images (just to see what we're working with)
def show_samples():
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))  # 2 rows, 5 columns
    for i, ax in enumerate(axes.flat):
        img, label = train_dataset[i]  # get image and its label
        ax.imshow(img.squeeze(), cmap='gray')  # squeeze removes extra dimension
        ax.set_title(f"Label: {label}")
        ax.axis('off')  # hide axis numbers
    plt.tight_layout()
    plt.savefig('mnist_samples.png')
    print("\nSaved sample images to 'mnist_samples.png'")

show_samples()

# now we make our neural networks!!
# this is the fun part

# first model - a simple feedforward network
# this one just flattens the image and passes it through layers
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()  # gotta call parent constructor
        self.flatten = nn.Flatten()  # turns 28x28 image into 784 long vector
        self.fc1 = nn.Linear(28*28, 128)  # first layer: 784 inputs -> 128 outputs
        self.fc2 = nn.Linear(128, 64)  # second layer: 128 -> 64
        self.fc3 = nn.Linear(64, 10)  # output layer: 64 -> 10 (for digits 0-9)
        self.relu = nn.ReLU()  # activation function (makes non-linear)
        self.dropout = nn.Dropout(0.2)  # randomly drops 20% of neurons (prevents overfitting)
    
    def forward(self, x):
        # this is where data flows through the network
        x = self.flatten(x)  # flatten image first
        x = self.relu(self.fc1(x))  # layer 1 + activation
        x = self.dropout(x)  # dropout for regularization
        x = self.relu(self.fc2(x))  # layer 2 + activation
        x = self.dropout(x)  # more dropout
        x = self.fc3(x)  # final layer (no activation here, loss function handles it)
        return x

# second model - convolutional neural network (CNN)
# this is better for images cuz it looks at local patterns
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # convolutional layers extract features from images
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 1 channel in, 32 out
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 32 in, 64 out
        self.pool = nn.MaxPool2d(2, 2)  # reduces size by half (28->14->7)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # after pooling twice, image is 7x7
        self.fc2 = nn.Linear(128, 10)  # output layer
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)  # slightly more dropout for cnn
    
    def forward(self, x):
        # pass through conv layers with pooling
        x = self.pool(self.relu(self.conv1(x)))  # conv1 -> relu -> pool
        x = self.pool(self.relu(self.conv2(x)))  # conv2 -> relu -> pool
        x = x.view(-1, 64 * 7 * 7)  # flatten for fully connected layers
        x = self.relu(self.fc1(x))  # fc layer
        x = self.dropout(x)
        x = self.fc2(x)  # output
        return x

# training function - this is where the learning happens!!
def train_model(model, train_loader, criterion, optimizer, epochs=5):
    model.train()  # puts model in training mode (enables dropout etc)
    train_losses = []  # keep track of losses
    train_accuracies = []  # keep track of accuracies
    
    # loop through each epoch (one epoch = one pass through entire dataset)
    for epoch in range(epochs):
        running_loss = 0.0  # total loss for this epoch
        correct = 0  # number of correct predictions
        total = 0  # total predictions made
        
        # tqdm makes a cool progress bar
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for images, labels in pbar:
            # move data to gpu if available
            images, labels = images.to(device), labels.to(device)
            
            # the training steps:
            optimizer.zero_grad()  # reset gradients (important!)
            outputs = model(images)  # forward pass (get predictions)
            loss = criterion(outputs, labels)  # calculate loss
            loss.backward()  # backward pass (calculate gradients)
            optimizer.step()  # update weights
            
            # track our progress
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)  # get the predicted class
            total += labels.size(0)
            correct += (predicted == labels).sum().item()  # count correct predictions
            
            # update progress bar with current stats
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 
                            'acc': f'{100*correct/total:.2f}%'})
        
        # calculate average loss and accuracy for this epoch
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        
        print(f'Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
    
    return train_losses, train_accuracies

# testing function - see how well our model does on new data
def test_model(model, test_loader):
    model.eval()  # puts model in evaluation mode (disables dropout)
    correct = 0
    total = 0
    all_preds = []  # save all predictions
    all_labels = []  # save all true labels
    
    # torch.no_grad() means we dont calculate gradients (saves memory, faster)
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)  # get predictions
            _, predicted = torch.max(outputs.data, 1)  # find the class with highest score
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())  # move to cpu for numpy
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100 * correct / total
    return accuracy, np.array(all_preds), np.array(all_labels)

# okay lets train the simple neural network first!
print("\n" + "="*50)
print("TRAINING SIMPLE NEURAL NETWORK")
print("="*50)

simple_model = SimpleNN().to(device)  # create model and move to gpu
criterion = nn.CrossEntropyLoss()  # loss function for classification
optimizer = optim.Adam(simple_model.parameters(), lr=0.001)  # adam optimizer is pretty good

# train it!
simple_losses, simple_accs = train_model(simple_model, train_loader, criterion, optimizer, epochs=5)
# test it!
simple_test_acc, _, _ = test_model(simple_model, test_loader)
print(f"\nSimple NN Test Accuracy: {simple_test_acc:.2f}%")

# now lets train the CNN (should be better!)
print("\n" + "="*50)
print("TRAINING CONVOLUTIONAL NEURAL NETWORK")
print("="*50)

conv_model = ConvNet().to(device)  # create cnn model
optimizer = optim.Adam(conv_model.parameters(), lr=0.001)  # new optimizer for new model

# train the cnn
conv_losses, conv_accs = train_model(conv_model, train_loader, criterion, optimizer, epochs=5)
# test the cnn
conv_test_acc, conv_preds, conv_labels = test_model(conv_model, test_loader)
print(f"\nConvNet Test Accuracy: {conv_test_acc:.2f}%")

# lets make some graphs to see how training went
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))  # 2 plots side by side

# plot the losses
ax1.plot(simple_losses, label='Simple NN', marker='o')
ax1.plot(conv_losses, label='ConvNet', marker='s')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training Loss')
ax1.legend()
ax1.grid(True)  # grid makes it easier to read

# plot the accuracies
ax2.plot(simple_accs, label='Simple NN', marker='o')
ax2.plot(conv_accs, label='ConvNet', marker='s')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Training Accuracy')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('training_progress.png')
print("\nSaved training progress to 'training_progress.png'")

# confusion matrix shows which digits get confused with each other
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(conv_labels, conv_preds)  # compare true vs predicted
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')  # annot shows numbers in boxes
plt.title('Confusion Matrix - ConvNet')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix.png')
print("Saved confusion matrix to 'confusion_matrix.png'")

# lets visualize some predictions to see what the model actually thinks
def visualize_predictions(model, dataset, num_images=10):
    model.eval()  # evaluation mode
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))  # 10 images total
    
    with torch.no_grad():  # dont need gradients for predictions
        for i, ax in enumerate(axes.flat):
            img, true_label = dataset[i]  # get test image
            img_tensor = img.unsqueeze(0).to(device)  # add batch dimension and move to gpu
            output = model(img_tensor)  # get prediction
            _, pred_label = torch.max(output, 1)  # get the predicted class
            
            # show the image
            ax.imshow(img.squeeze(), cmap='gray')
            # green if correct, red if wrong
            color = 'green' if pred_label.item() == true_label else 'red'
            ax.set_title(f'True: {true_label}\nPred: {pred_label.item()}', color=color)
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('predictions.png')
    print("Saved predictions to 'predictions.png'")

visualize_predictions(conv_model, test_dataset)

# print final summary
print("\n" + "="*50)
print("FINAL RESULTS")
print("="*50)
print(f"Simple Neural Network Test Accuracy: {simple_test_acc:.2f}%")
print(f"Convolutional Neural Network Test Accuracy: {conv_test_acc:.2f}%")
print(f"\nBest Model: {'ConvNet' if conv_test_acc > simple_test_acc else 'Simple NN'}")

# save the models so we can use them later without retraining
torch.save(simple_model.state_dict(), 'simple_nn_model.pth')
torch.save(conv_model.state_dict(), 'convnet_model.pth')
print("\nModels saved:")
print("- simple_nn_model.pth")
print("- convnet_model.pth")

print("\n" + "="*50)
print("PROJECT COMPLETE!")
print("="*50)
