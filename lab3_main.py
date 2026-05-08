import numpy as np
from tqdm import tqdm
from collections import defaultdict

from lab1_proto import mfcc, mspec

from lab2_proto import *
from lab2_tools import *
from prondict import prondict

from lab3_tools import *
from lab3_proto import *




### SECTION 4.1

# build ordered list of HMM states for DNN output classes
# -> 3 x nr_phones x nr_emit_states output classes (+1 for short pause)
phoneHMMs = np.load('../Lab2-DT2119/lab2_models_all.npz', allow_pickle=True)['phoneHMMs'].item()
phones = sorted(phoneHMMs.keys())
nstates = {phone: phoneHMMs[phone]['means'].shape[0] for phone in phones}   
stateList = [ph + '_' + str(idx) for ph in phones for idx in range(nstates[ph])]
np.savez('stateList.npz', stateList=stateList)


### SECTION 4.2

filename = 'tidigits/disc_4.1.1/tidigits/train/man/nw/z43a.wav'
samples, samplingrate = loadAudio(filename)
lmfcc = mfcc(samples)

# transcribe character-level transcription into phoneme-level transcription
wordTrans = list(path2info(filename)[2])
phoneTrans = words2phones(wordTrans, prondict)

# result confirmation
example = np.load('lab3_example.npz', allow_pickle=True)['example'].item()
viterbi_states = forcedAlignment(example['lmfcc'], phoneHMMs, example['phoneTrans'])
trans = frames2trans(viterbi_states, outfilename='z43a.lab')
trans_to_audacity(trans, 'z43a.txt') 
print('Match?', viterbi_states == example['viterbiStateTrans'])


### SECTION 4.3

def extract_set(root_dir):
    # First, collect the file list so tqdm can show ETA
    wav_files = []
    for root, dirs, files in os.walk(root_dir):
        for f in files:
            if f.endswith('.wav'):
                wav_files.append(os.path.join(root, f))

    data = []
    for filename in tqdm(wav_files):
        samples, _ = loadAudio(filename)

        # both feature types from the same audio
        mspec_feats = mspec(samples)             # shape (N, 40) according to lab 1
        lmfcc_feats = mfcc(samples)              # shape (N, 13) according to lab 1

        # phone-level transcription from corresponding filename
        wordTrans = list(path2info(filename)[2])
        phoneTrans = words2phones(wordTrans, prondict)

        # forced alignment via Viterbi (length = number of frames in utterance)
        viterbiStateTrans = forcedAlignment(lmfcc_feats, phoneHMMs, phoneTrans)

        # conversion of state labels to integer indices
        state_to_idx = {s: i for i, s in enumerate(stateList)}
        targets = np.array([state_to_idx[s] for s in viterbiStateTrans], dtype=np.int32)

        data.append({
            'filename': filename,
            'lmfcc':    lmfcc_feats,
            'mspec':    mspec_feats,
            'targets':  targets,
        })
    return data

# traindata = extract_set('tidigits/disc_4.1.1/tidigits/train')   # around 8600 utts, 1.5M frames
# testdata  = extract_set('tidigits/disc_4.2.1/tidigits/test')    # 8700 utts, 0.5M frames
# np.savez('traindata.npz', traindata=traindata, stateList=stateList)
# np.savez('testdata.npz',  testdata=testdata, stateList=stateList)
traindata = np.load('traindata.npz', allow_pickle=True)['traindata']
testdata = np.load('testdata.npz', allow_pickle=True)['testdata']


### SECTION 4.4

# group utterance indices by gender and speaker (least to most specific)
groups = defaultdict(list)
for i, utt in enumerate(traindata):
    gender, speaker, _, _ = path2info(utt['filename'])
    groups[(gender, speaker)].append(i)

# get the unique speakers for each gender
men_speakers   = sorted(s for (g, s) in groups if g == 'man')
women_speakers = sorted(s for (g, s) in groups if g == 'woman')

print(f"{len(men_speakers)} male speakers, {len(women_speakers)} female speakers")

# picking around 10% of each gender ensures gender stratification for the validation/test set
rng = np.random.RandomState(42)
rng.shuffle(men_speakers)
rng.shuffle(women_speakers)

n_val_men   = round(0.10 * len(men_speakers))
n_val_women = round(0.10 * len(women_speakers))

val_speakers = (
    {('man',   s) for s in men_speakers[:n_val_men]} |
    {('woman', s) for s in women_speakers[:n_val_women]}
)

# store the utterances in respective dataset split
train_idx, val_idx = [], []
for key, idxs in groups.items():
    (val_idx if key in val_speakers else train_idx).extend(idxs)

train_split = [traindata[i] for i in train_idx]
val_split   = [traindata[i] for i in val_idx]


### SECTION 4.5

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

features = train_split[0]['lmfcc']
frame_idx = np.random.randint(0, len(features)) 
stack = stackfeatures(frame_idx, features)
print(f"Time step: {frame_idx}. LMFCC features: {features[frame_idx]}. Stacked: {stack}")
