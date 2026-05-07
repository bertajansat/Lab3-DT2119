import os
from lab1_proto import mfcc, mspec
from lab2_proto import viterbi, concatHMMs
from lab2_tools import log_multivariate_normal_density_diag
import numpy as np
from lab3_tools import *
from prondict import prondict
from tqdm import tqdm
import random
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
import torch

def words2phones(wordList, pronDict, addSilence=True, addShortPause=True):
    """ word2phones: converts word level to phone level transcription adding silence

    Args:
       wordList: list of word symbols
       pronDict: pronunciation dictionary. The keys correspond to words in wordList
       addSilence: if True, add initial and final silence
       addShortPause: if True, add short pause model "sp" at end of each word
    Output:
       list of phone symbols
    """
    phoneList = []
    if addSilence:
       phoneList.append('sil')

    for i, word in enumerate(wordList):
       phoneList.extend(pronDict[word])
       if addShortPause and i < len(wordList) - 1:
          phoneList.append('sp')
    if addSilence:
       phoneList.append('sil')

    return phoneList

def forcedAlignment(lmfcc, phoneHMMs, phoneTrans):
    """ forcedAlignmen: aligns a phonetic transcription at the state level

    Args:
       lmfcc: NxD array of MFCC feature vectors (N vectors of dimension D)
              computed the same way as for the training of phoneHMMs
       phoneHMMs: set of phonetic Gaussian HMM models
       phoneTrans: list of phonetic symbols to be aligned including initial and
                   final silence

    Returns:
       list of strings in the form phoneme_index specifying, for each time step
       the state from phoneHMMs corresponding to the viterbi path.
    """
    utteranceHMM = concatHMMs(phoneHMMs, phoneTrans) # Concatenate HMMs to create specific HMM for utterance (phrase)
    stateTrans = [phone + '_' + str(stateid) for phone in phoneTrans  #  For each state in utteranceHMM, obtain corresponding unique state identifier 
            for stateid in range(nstates[phone])]

    lpr = log_multivariate_normal_density_diag(lmfcc,utteranceHMM['means'],utteranceHMM['covars'])
    M = lpr.shape[1]
    viterbi_loglik, viterbi_path = viterbi(lpr,np.log(utteranceHMM['startprob'][:M]),np.log(utteranceHMM['transmat'][:M, :M]))

    # Convert viterbi path to unique state names in stateList:
    viterbiStateTrans = [stateTrans[i] for i in viterbi_path]

    return viterbiStateTrans

# 4.5. Acoustic Context (Dynamic Features)
def stackfeatures(n, features):
    T = len(features)
    stacked = np.zeros((7, len(features[0])))

    if n >= 3 and n <= T - 4:
        for i in range(7):
            stacked[i] = features[n - 3 + i]

    else: # Special cases (mirroring)
        for i in range(7):
            idx = n - 3 + i

            # mirror at boundaries
            if idx < 0:
                idx = -idx
            if idx >= T:
                idx = 2*T - idx - 2

            stacked[i] = features[idx]
    return stacked


# 4.1 Target Class Definition

phoneHMMs = np.load('../lab2/lab2_models_all.npz',allow_pickle=True)['phoneHMMs'].item()
phones = sorted(phoneHMMs.keys())
nstates = {phone: phoneHMMs[phone]['means'].shape[0] for phone in phones}
stateList = [ph + '_' + str(id) for ph in phones for id in range(nstates[ph])]

# Recover the numerical index of a particular state in the list
stateList.index('ay_2')

# 4.2. Forced Alignment

# Only for one file

filename = 'tidigits/disc_4.1.1/tidigits/train/man/nw/z43a.wav'
samples, samplingrate = loadAudio(filename)
lmfcc = mfcc(samples) 


wordTrans = list(path2info(filename)[2])  # Obtain word level transcription  
print(wordTrans)
phoneTrans = words2phones(wordTrans, prondict) # Obtain phone level transcription given words in file + phonetic dictionari
print(phoneTrans)
#utteranceHMM = concatHMMs(phoneHMMs, phoneTrans) # Concatenate HMMs to create specific HMM for utterance (phrase)

#stateTrans = [phone + '_' + str(stateid) for phone in phoneTrans  #  For each state in utteranceHMM, obtain corresponding unique state identifier 
#            for stateid in range(nstates[phone])]

viterbiStateTrans = forcedAlignment(lmfcc, phoneHMMs, phoneTrans)
# Save transcription on z43a.lab (visualize to check that it did it correctly)
frames2trans(viterbiStateTrans, outfilename='z43a.lab')

# 4.3 FEATURE EXTRACTION
"""
# Train data:

# Progress bar:
total = sum(len([f for f in files if f.endswith('.wav')]) 
            for _, _, files in os.walk('tidigits/disc_4.1.1/tidigits/train'))

pbar = tqdm(total=total, desc="Feature extraction")


traindata = []
for root, dirs, files in os.walk('tidigits/disc_4.1.1/tidigits/train'):
    for file in files:
      if file.endswith('.wav'):
         filename = os.path.join(root, file)
         samples, samplingrate = loadAudio(filename)
         mspec_feat = mspec(samples)
         lmfcc = mfcc(samples) 
         wordTrans = list(path2info(filename)[2])  # Obtain word level transcription  
         phoneTrans = words2phones(wordTrans, prondict) # Obtain phone level transcription given words in file + phonetic dictionari
         
         targets = forcedAlignment(lmfcc, phoneHMMs, phoneTrans)

         # Convert viterbi path to unique state names in stateList:
         #targets = [stateList.index(stateTrans[i]) for i in viterbi_path] 
         traindata.append({'filename': filename, 'lmfcc': lmfcc,'mspec': mspec_feat, 'targets': targets})
         pbar.update(1) # Progress bar

pbar.close() # Close progress bar 
np.savez('traindata.npz', traindata=traindata)


# Test data:

# Progress bar:
total = sum(len([f for f in files if f.endswith('.wav')]) 
            for _, _, files in os.walk('tidigits/disc_4.2.1/tidigits/test'))

pbar = tqdm(total=total, desc="Feature extraction")


testdata = []
for root, dirs, files in os.walk('tidigits/disc_4.2.1/tidigits/test'):
    for file in files:
      if file.endswith('.wav'):
         filename = os.path.join(root, file)
         samples, samplingrate = loadAudio(filename)
         lmfcc = mfcc(samples) 
         mspec_feat = mspec(samples)
         wordTrans = list(path2info(filename)[2])  # Obtain word level transcription  
         phoneTrans = words2phones(wordTrans, prondict) # Obtain phone level transcription given words in file + phonetic dictionari
         
         targets = forcedAlignment(lmfcc, phoneHMMs, phoneTrans)

         # Convert viterbi path to unique state names in stateList:
         #targets = [stateList.index(stateTrans[i]) for i in viterbi_path] 
         testdata.append({'filename': filename, 'lmfcc': lmfcc,'mspec': mspec_feat, 'targets': targets})
         pbar.update(1) # Progress bar

pbar.close() # Close progress bar 
np.savez('testdata.npz', testdata=testdata)
"""

# 4.4 Training and Validation Sets

def train_val_sets(data):
   speaker_dict = {}
   for utterance in data:

      gender, speaker, digits, repetition = path2info(utterance['filename'])
      if speaker not in speaker_dict:
         speaker_dict[speaker] = {'gender': gender, 'utterances': [utterance]}
      else:
         speaker_dict[speaker]['utterances'].append(utterance)   # Save utterances of same speaker together


   # Divide by gender
   male = sorted(s for s in speaker_dict if speaker_dict[s]['gender'] == 'man')  # Keep only male speakers
   female = sorted(s for s in speaker_dict if speaker_dict[s]['gender'] == 'woman') # Keep only female speakers

   # Speakers might be in order, shuffle to remove bias:
   rng = np.random.RandomState(42) # seed
   rng.shuffle(male) 
   rng.shuffle(female)

   # Split male and female datasets:
   cut_male = int(0.9 * len(male))
   male_train = male[:cut_male]
   male_val = male[cut_male:]

   cut_fem = int(0.9 * len(female))
   female_train = female[:cut_fem]
   female_val = female[cut_fem:]

   train_set = male_train + female_train
   val_set = male_val + female_val

   # Return datasets
   train_data = []
   for speaker in train_set:
      train_data += speaker_dict[speaker]['utterances']
   val_data = []
   for speaker in val_set:
      val_data += speaker_dict[speaker]['utterances']

   return train_set, val_set, train_data, val_data

data = np.load('traindata.npz', allow_pickle=True)['traindata']
"""
speaker_dict = {}

for utterance in data:

   gender, speaker, digits, repetition = path2info(utterance['filename'])
   if speaker not in speaker_dict:
      speaker_dict[speaker] = {'gender': gender, 'utterances': [utterance]}
   else:
      speaker_dict[speaker]['utterances'].append(utterance)   # Save utterances of same speaker together


# Divide by gender
male = [s for s in speaker_dict if speaker_dict[s]['gender'] == 'man']  # Keep only male speakers
female = [s for s in speaker_dict if speaker_dict[s]['gender'] == 'woman'] # Keep only female speakers

# Speakers might be in order, shuffle to remove bias:
random.shuffle(male) 
random.shuffle(female)

# Split male and female datasets:
cut_male = int(0.9 * len(male))
male_train = male[:cut_male]
male_val = male[cut_male:]

cut_fem = int(0.9 * len(female))
female_train = female[:cut_fem]
female_val = female[cut_fem:]

train_set = male_train + female_train
val_set = male_val + female_val
"""

train_set, val_set, train_data, val_data = train_val_sets(data)

print(f"\nTrain set size: {len(train_set)}.") # Women: {len(female_train)}. Men: {len(male_train)}.")
print(f"Val set size: {len(val_set)}.") #Women: {len(female_val)}. Men: {len(male_val)}")

# 4.5. Acoustic Context (Dynamic Features)

# Test:
#features = data[0]['lmfcc']
#n = np.random.randint(0, len(features)) 
#stack = stackfeatures(n, features)
#print(f"n: {n}. Features n: {features[n]}. Stacked: {stack}")

# 4.6. Feature Standardisation

testdata = np.load('testdata.npz', allow_pickle=True)['testdata']

# Collect al coefficients for training data

def flatten_data(data, feature='lmfcc',dynamic_f=False):
   features = []
   targets = []
   for utterance in data:
      if feature == 'lmfcc':
        if dynamic_f:
            feat = []
            T = len(utterance['lmfcc'])
            for t in range(T):
                stacked = stackfeatures(t, utterance['lmfcc']).reshape(-1)  # (7*13,)
                feat.append(stacked)
            feat = np.vstack(feat)   # (T, 7*13)
        else:
            feat = utterance['lmfcc']
    
        features.append(feat)     # (T, D)
      elif feature == 'mspec':
        if dynamic_f:
            feat = []
            T = len(utterance['mspec'])
            for t in range(T):
                stacked = stackfeatures(t, utterance['mspec']).reshape(-1)  # (7*13,)
                feat.append(stacked)
            feat = np.vstack(feat)   # (T, 7*13)
        else:
            feat = utterance['mspec']
        features.append(feat)     # (T, D)
      else:
         print("\nIncorrect feature type.")
      targets.append(utterance['targets'])   # (T,)

   # Concatenate  --> TODO: explain on notes
   X = np.vstack(features)      # (N, D)
   y = np.concatenate(targets) # (N,)

   return X, y

def feature_standarization(training_set, val_set, test_set, stateList, feature='lmfcc',dynamic_f=False):
   if feature == 'lmfcc':
      X_train, y_train = flatten_data(training_set,dynamic_f=dynamic_f)
      X_val, y_val = flatten_data(val_set,dynamic_f=dynamic_f)
      X_test, y_test = flatten_data(test_set,dynamic_f=dynamic_f)
   elif feature == 'mspec':
      X_train, y_train = flatten_data(training_set, feature='mspec',dynamic_f=dynamic_f)
      X_val, y_val = flatten_data(val_set, feature='mspec',dynamic_f=dynamic_f)
      X_test, y_test = flatten_data(test_set, feature='mspec',dynamic_f=dynamic_f)
   else:
      print("\nIncorrect feature type.")
      return

   # Normalization coefficients
   scaler = StandardScaler()
   scaler.fit(X_train)

   # Normalization:

   X_train = scaler.transform(X_train)
   X_val   = scaler.transform(X_val)
   X_test  = scaler.transform(X_test)

   print("TRAIN:", X_train.shape, y_train.shape)
   print("VAL:  ", X_val.shape,   y_val.shape)
   print("TEST: ", X_test.shape,  y_test.shape)

   # Convert feature arrays to 32 bits floating point
   X_train = X_train.astype('float32') # lmfcc_train_x
   X_val = X_val.astype('float32')
   X_test = X_test.astype('float32')

   # Convert target arrays into a one-hot encoding:
   output_dim = len(stateList)
   state_to_idx = {state: i for i, state in enumerate(stateList)}
   y_train = np.array([state_to_idx[s] for s in y_train])
   y_val   = np.array([state_to_idx[s] for s in y_val])
   y_test  = np.array([state_to_idx[s] for s in y_test])

   y_train = F.one_hot(torch.tensor(y_train),num_classes=output_dim).float()  # Change if necessary TODO
   y_val = F.one_hot(torch.tensor(y_val),num_classes=output_dim).float()
   y_test = F.one_hot(torch.tensor(y_test),num_classes=output_dim).float()

   return X_train, X_val, X_test, y_train, y_val, y_test, state_to_idx

print("\nLMFCC:")
X_train, X_val, X_test, y_train, y_val, y_test, state_to_idx = feature_standarization(train_data, val_data, testdata, stateList, feature='lmfcc')
print("\nFilterbank:")
X_train_mspec, X_val_mspec, X_test_mspec, y_train_mspec, y_val_mspec, y_test_mspec, state_to_idx = feature_standarization(train_data, val_data, testdata, stateList, feature='mspec')

"""


train_features = []
train_targets = []

for speaker in train_set:
    for utterance in speaker_dict[speaker]['utterances']:
      train_features.append(utterance['lmfcc'])     # (T, D)
      train_targets.append(utterance['targets'])   # (T,)

# Concatenate  --> TODO: explain on notes
X_train = np.vstack(train_features)      # (N, D)
y_train = np.concatenate(train_targets) # (N,)

# Validation set concatenation
val_features = []
val_targets = []

for speaker in val_set:
    for utterance in speaker_dict[speaker]['utterances']:
        val_features.append(utterance['lmfcc'])
        val_targets.append(utterance['targets'])

X_val = np.vstack(val_features)
y_val = np.concatenate(val_targets)

# Test set concatenation
test_features = []
test_targets = []

for utterance in testdata:
    test_features.append(utterance['lmfcc'])
    test_targets.append(utterance['targets'])

X_test = np.vstack(test_features)
y_test = np.concatenate(test_targets)



# Normalization coefficients

scaler = StandardScaler()
scaler.fit(X_train)


# Normalization:

X_train = scaler.transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

print("TRAIN:", X_train.shape, y_train.shape)
print("VAL:  ", X_val.shape,   y_val.shape)
print("TEST: ", X_test.shape,  y_test.shape)

# Convert feature arrays to 32 bits floating point
X_train = X_train.astype('float32') # lmfcc_train_x
X_val = X_val.astype('float32')
X_test = X_test.astype('float32')

# Convert target arrays into a one-hot encoding:
output_dim = len(stateList)
state_to_idx = {state: i for i, state in enumerate(stateList)}
y_train = np.array([state_to_idx[s] for s in y_train])
y_val   = np.array([state_to_idx[s] for s in y_val])
y_test  = np.array([state_to_idx[s] for s in y_test])

y_train = F.one_hot(torch.tensor(y_train),num_classes=output_dim)
y_val = F.one_hot(torch.tensor(y_val),num_classes=output_dim)
y_test = F.one_hot(torch.tensor(y_test),num_classes=output_dim)
"""
## 5. Phoneme Recognition with Deep Neural Networks

#print(output_dim)