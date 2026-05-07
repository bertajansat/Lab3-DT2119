
# This file contains boiler-plate code for defining and training a network in PyTorch.
# Please see PyTorch documentation and tutorials for more information 
# e.g. https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
# IMPORTS:
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from lab3_proto import feature_standarization, stackfeatures, train_val_sets
import numpy as np
from sklearn.metrics import confusion_matrix
from nltk.metrics.distance import edit_distance
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Variable and device definitions:
audio_feature = 'mspec'
dynamic_features = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Input size depending on audio features:
if audio_feature == 'mspec' and dynamic_features == False:
    input_size=40
elif audio_feature == 'mspec' and dynamic_features == True:
    input_size=280 
elif audio_feature == 'lmfcc' and dynamic_features == False:
    input_size=13
else:
    input_size = 61

# Neural network architecture
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Simple: 3 layers
        self.fc1 = nn.Linear(input_size, 256)   # input → hidden  # 13 without dynamic_features
        self.fc2 = nn.Linear(256, 256)  # hidden → hidden
        self.fc3 = nn.Linear(256, 61)   # hidden → output

    def forward(self, x):
        # hidden layers with ReLU
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # output layer
        x = self.fc3(x)

        # softmax (output activation)
        #x = F.softmax(x, dim=1)  --> Cross Entropy loss already applies it
        return x

def count_parameters(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)

# instantiate the network and print the structure
net = Net().to(device)
print(net)
print(f'number of prameters:{count_parameters(net)}')

# define your loss criterion (see https://pytorch.org/docs/stable/nn.html#loss-functions) # TODO
#criterion = nn.BCEWithLogitsLoss()
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
    train_x, val_x, test_x, train_y, val_y, test_y, state_to_idx = feature_standarization(train_dataset, val_dataset, test_dataset, stateList, feature='lmfcc',dynamic_f=dynamic_features)
else: 
    train_x, val_x, test_x, train_y, val_y, test_y, state_to_idx = feature_standarization(train_dataset, val_dataset, test_dataset, stateList, feature='mspec',dynamic_f=dynamic_features)

train_x = torch.tensor(train_x, dtype=torch.float32)
val_x = torch.tensor(val_x, dtype=torch.float32)
test_x = torch.tensor(test_x, dtype=torch.float32)



batch_size = 64

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
num_epochs = 10


for epoch in range(num_epochs):
    net.train()
    train_loss = 0.0
    val_correct = 0
    val_total = 0
    train_correct = 0
    train_total = 0
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}"): # Add progress bar
        inputs = inputs.to(device)
        labels = labels.to(device)
        
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
            inputs = inputs.to(device)
            labels = labels.to(device)
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


# save the trained network
torch.save(net.state_dict(), 'trained-net.pt')

# finally evaluate model on the test set here: 
net.eval()
test_x = test_x.to(device)
outputs = net(test_x)
y_pred = torch.argmax(outputs, dim=1) # convert predictions into class indices (class with highest probability)
y_true = torch.argmax(test_y, dim=1) # convert one-hot vectors back into class indices
idx_to_state = {i: state for state, i in state_to_idx.items()}
true_states = [idx_to_state[i.item()] for i in y_true]
pred_states = [idx_to_state[i.item()] for i in y_pred]


def eval_frame_state(pred_states, true_states):
    accuracy = np.mean(np.array(true_states == np.array(pred_states)))
    cm = confusion_matrix(true_states, pred_states)
    return accuracy, cm

def eval_frame_phoneme(pred_states, true_states):
    # Remove _x (keep only phoneme, not phoneme state)
    true_phonemes = [s.split('_')[0] for s in true_states]
    pred_phonemes = [s.split('_')[0] for s in pred_states]
    
    accuracy = np.mean(np.array(true_phonemes) == np.array(pred_phonemes))
    cm = confusion_matrix(true_phonemes, pred_phonemes)
    return accuracy, cm

def collapse_repeats(sequence):  # Remove repeated states (from ox_0, ox_0, ox_1, ox_2, ox_2 --> ox_0, ox_1, ox_2)
    collapsed = [sequence[0]]
    for s in sequence[1:]:
        if s != collapsed[-1]:
            collapsed.append(s)
    return collapsed

def eval_dist_state(pred_states, true_states):
    total_dist = 0
    total_len = 0

    for pred_seq, true_seq in zip(pred_states, true_states):

        pred_c = collapse_repeats(pred_seq)
        true_c = collapse_repeats(true_seq)

        dist = edit_distance(true_c, pred_c)

        total_dist += dist
        total_len += len(true_c)

    per = total_dist / total_len if total_len > 0 else 0

    return per, total_dist

def eval_dist_phoneme(pred_states, true_states):
    true_phonemes = [s.split('_')[0] for s in true_states]
    pred_phonemes = [s.split('_')[0] for s in pred_states]

    total_dist = 0
    total_len = 0

    for pred_seq, true_seq in zip(pred_phonemes, true_phonemes):

        pred_c = collapse_repeats(pred_seq)
        true_c = collapse_repeats(true_seq)

        dist = edit_distance(true_c, pred_c)

        total_dist += dist
        total_len += len(true_c)

    per = total_dist / total_len if total_len > 0 else 0

    #true_collapsed = collapse_repeats(true_phonemes)
    #pred_collapsed = collapse_repeats(pred_phonemes)

    #dist = edit_distance(true_collapsed, pred_collapsed)
    #per = dist / len(true_collapsed)

    return per, total_dist
    
print("Starting eval")

# 1. Frame-level state evaluation
print("Frame-level state evaluation")
state_acc, state_cm = eval_frame_state(pred_states, true_states)

# 2. Frame-level phoneme evaluation
print("Frame-level phoneme evaluation")
phoneme_acc, phoneme_cm = eval_frame_phoneme(pred_states, true_states)

# 3. Edit distance at state level
print("Edit distance at state level")
state_per, state_dist = eval_dist_state(pred_states, true_states)

# 4. Edit distance at phoneme level
print("Edit distance at phoneme level")
phoneme_per, phoneme_dist = eval_dist_phoneme(pred_states, true_states)

# ---------------------------------------------------
# Print results
# ---------------------------------------------------

print("\n===== EVALUATION RESULTS =====\n")

print(f"Frame accuracy (state level):     {state_acc:.4f}")
print(f"Frame accuracy (phoneme level):  {phoneme_acc:.4f}")

print(f"\nState-level edit distance:       {state_dist}")
print(f"State-level PER:                 {state_per:.4f}")

print(f"\nPhoneme-level edit distance:     {phoneme_dist}")
print(f"Phoneme-level PER:               {phoneme_per:.4f}")

# ---------------------------------------------------
# Save confusion matrices
# ---------------------------------------------------

# State labels
state_labels = sorted(list(set(true_states)))

plt.figure(figsize=(12, 10))
sns.heatmap(
    state_cm,
    xticklabels=state_labels,
    yticklabels=state_labels,
    cmap="plasma"
)

plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("State-level Confusion Matrix")

plt.tight_layout()
plt.savefig("confusion_matrix_states_mspec_dyn.png", dpi=300)
plt.close()

# Phoneme level confusion matrix
true_phonemes = [s.split('_')[0] for s in true_states]
phoneme_labels = sorted(list(set(true_phonemes)))

plt.figure(figsize=(12, 10))
sns.heatmap(
    phoneme_cm,
    xticklabels=phoneme_labels,
    yticklabels=phoneme_labels,
    cmap="plasma"
)

plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Phoneme-level Confusion Matrix")

plt.tight_layout()
plt.savefig("confusion_matrix_phonemes_mspec_dyn.png", dpi=300)
plt.close()


