
# This file contains boiler-plate code for defining and training a network in PyTorch.
# Please see PyTorch documentation and tutorials for more information 
# e.g. https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html

import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from lab3_proto import feature_standarization, stackfeatures, train_val_sets
import numpy as np

audio_feature = 'lmfcc'


# define the neural network architecture
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Simple: 3 layers
        self.fc1 = nn.Linear(13, 256)   # input → hidden
        self.fc2 = nn.Linear(256, 256)  # hidden → hidden
        self.fc3 = nn.Linear(256, 61)   # hidden → output

    def forward(self, x):
        # hidden layers with ReLU
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # output layer
        x = self.fc3(x)

        # softmax (output activation)
        x = F.softmax(x, dim=1)
        return x

def count_parameters(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)

# instantiate the network and print the structure
net = Net()
print(net)
print(f'number of prameters:{count_parameters(net)}')

# define your loss criterion (see https://pytorch.org/docs/stable/nn.html#loss-functions) # TODO
criterion = torch.nn.CrossEntropyLoss()

# define the optimizer 
optimizer = torch.optim.Adam(net.parameters())

# prepare/load the data into tensors  
phoneHMMs = np.load('../lab2/lab2_models_all.npz',allow_pickle=True)['phoneHMMs'].item()
phones = sorted(phoneHMMs.keys())
nstates = {phone: phoneHMMs[phone]['means'].shape[0] for phone in phones}
stateList = [ph + '_' + str(id) for ph in phones for id in range(nstates[ph])]  # State list obtention


data = np.load('traindata.npz', allow_pickle=True)['traindata'] #Load data
train_set, val_set, train_dataset, val_dataset = train_val_sets(data) # Divide into train and validation

test_dataset = np.load('testdata.npz', allow_pickle=True)['testdata']
if audio_feature == 'lmfcc':
    train_x, val_x, test_x, train_y, val_y, test_y = feature_standarization(train_dataset, val_dataset, test_dataset, stateList, feature='lmfcc')
else: 
    train_x, val_x, test_x, train_y, val_y, test_y = feature_standarization(train_dataset, val_dataset, test_dataset, stateList, feature='mspec')

train_x = torch.tensor(train_x, dtype=torch.float32)
val_x = torch.tensor(val_x, dtype=torch.float32)
test_x = torch.tensor(test_x, dtype=torch.float32)



batch_size = 64 #256

# create the data loaders for training and validation sets
print(type(train_x))
print(type(train_y))

train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = torch.utils.data.TensorDataset(val_x, val_y)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# setup logging so that you can follow training using TensorBoard (see https://pytorch.org/docs/stable/tensorboard.html)
writer = SummaryWriter("runs/lab3_experiment")

# train the network
num_epochs = 50


for epoch in range(num_epochs):
    net.train()
    train_loss = 0.0
    val_correct = 0
    val_total = 0
    train_correct = 0
    train_total = 0
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}"): # Add progress bar
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # accumulate the training loss
        train_loss += loss.item()

        # Accuracy
        preds = torch.argmax(outputs, dim=1)
        true = torch.argmax(labels, dim=1)

        train_correct += (preds == true).sum().item()
        train_total += labels.size(0)

    # calculate the validation loss
    net.eval()
    with torch.no_grad():
        val_loss = 0.0
        for inputs, labels in tqdm(val_loader, desc="Validation", leave=False):
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # Accuracy computation:
            preds = torch.argmax(outputs, dim=1)
            true = torch.argmax(labels, dim=1)

            val_correct += (preds == true).sum().item()
            val_total += labels.size(0)

    # print the epoch loss and accuracy:
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    train_acc = train_correct / train_total
    val_acc = val_correct / val_total

    print(f'Epoch {epoch}: train_loss={train_loss}, train_accuracy={train_acc}, val_loss={val_loss}, val_accuracy={val_acc},')
    writer.add_scalars('Loss', {
        'train': train_loss,
        'val': val_loss
    }, epoch)

    writer.add_scalars('Accuracy', {
        'train': train_acc,
        'val': val_acc
    }, epoch)

# finally evaluate model on the test set here: TODO later
#outputs = net(X_test)
#y_pred = torch.argmax(outputs, dim=1) # convert predictions into class indices (class with highest probability)
#y_true = torch.argmax(y_test, dim=1) # convert one-hot vectors back into class indices

# save the trained network
torch.save(net.state_dict(), 'trained-net.pt')
