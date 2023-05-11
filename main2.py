import os
import music21
import librosa
import numpy as np
from chord_extractor.extractors import Chordino


chordino = Chordino(roll_on=1)


# Run extraction
chords = chordino.extract('Jazz_Standards_Data/Wav_Files/26-2.wav')



# Set the path to the directory containing the .wav files
directory = 'Jazz_Standards_Data/Wav_Files'

# Get a list of all .wav files in the directory
wav_files = [f for f in os.listdir(directory) if f.endswith('.wav')]

# Sort the list of filenames alphabetically
wav_files_sorted = sorted(wav_files)

# Loop through the sorted list of filenames


directory2 = 'Jazz_Standards_Data/Xml_Files'

# Get a list of all .wav files in the directory
xml_files = [f for f in os.listdir(directory2) if f.endswith('.xml')]

# Sort the list of filenames alphabetically
xml_files_sorted = sorted(xml_files)


# Load lead sheet file


def extract_info(filename):
    lead_sheet = music21.converter.parse(filename)
    import xml.etree.ElementTree as ET

    # parse the XML file
    tree = ET.parse(filename)

    # get the root element
    root = tree.getroot()

    # find all the <harmony> elements
    harmony_elements = root.findall('.//harmony')
    note_elements = root.findall('.//note')
    divisions1 = root.findall('.//attributes/divisions')
    keys1 = root.findall('.//key')
    divisions = []
    durations = []
    notes = []
    keys = []

    # loop through the <harmony> elements and extract the chord symbol and root note
    for harmony_element in harmony_elements:
        chord_symbol = harmony_element.find('kind').text
        root_note = harmony_element.find('root').find('root-step').text
        notes.append(chord_symbol)

    for note_element in note_elements:
        duration = note_element.find('duration').text
        durations.append(duration)

    for division in divisions1:
        divisions.append(division.text)

    for key in keys1:
        keys.append(key.find('fifths').text)

    return notes, durations, divisions, keys


# Loop through the sorted list of filenames

all_chords = []
all_durations = []
all_divisions = []
file_names = []
all_keys = []
for filename in xml_files_sorted:
    file_path = os.path.join(directory2, filename)
    file_names.append(file_path)
    chords, durations, divisions, keys = extract_info(file_path)

    all_divisions.append(divisions)
    all_chords.append(np.array(chords))
    all_durations.append(np.array(durations, dtype=np.int64))
    all_keys.append(keys)

import matplotlib.pyplot as plt

all_durations = np.array(all_durations, dtype=object)

def generate_spectrograms(wav_files_sorted):
    spec_array = []
    spec_db_array = []
    sr_array = []

    for filename in wav_files_sorted:
        wav_file = "Jazz_Standards_Data/Wav_Files/" + filename
        # Compute spectrogram
        y, sr = librosa.load(wav_file)
        sr_array.append(sr)
        spec = np.abs(librosa.stft(y))
        spec_db = librosa.amplitude_to_db(spec, ref=np.max)

        spec_array.append(spec)
        spec_db_array.append(spec_db)

    return spec_array, spec_db_array, sr_array


# Generate spectrograms
# print(len(spec_db_array))
# Save spectrograms as images
i = 0
for spec_db in spec_db_array:
    print(i)

    plt.figure(figsize=(10, 6))
    librosa.display.specshow(spec_db, sr=sr_array[i], x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Spectrogram {i + 1}')
    plt.savefig(f'Spectrograms/spectrogram_{i + 1}.png')  # Save plot as image
    plt.close()  # Close plot
    i += 1


i = 0
for filename in wav_files_sorted:
    wav_files_sorted[i] = "Jazz_Standards_Data/Wav_Files/" + filename
    i += 1

wav_files_sorted = np.array(wav_files_sorted)
# all_chords = np.array(all_chords)
# all_durations = np.array(all_durations)

from sklearn.model_selection import train_test_split

import torch
import torchvision
import torchvision.transforms as transforms

from PIL import Image

spectrograms = []
i = 1
while i < len(spec_array):
    file = "spectrogram_" + str(i) + ".png"
    i += 1
    file_path = "Spectrograms/" + file
    image = Image.open(file_path)
    spectrogram = np.array(image)

    spectrogram = np.array(image)

    spectrograms.append(spectrogram)
spectrograms = np.array(spectrograms)


spectrograms_train, spectrograms_test, chord_names_train, chord_names_test, chord_durations_train, chord_durations_test = train_test_split(
    spectrograms, all_chords, all_durations, test_size=0.2, random_state=48)

all_chords2 = []
for chord in chord_names_train:
    for c in chord:
        all_chords2.append(c)
all_chords2 = np.array(all_chords2)
unique_chords = np.unique(all_chords2)  # Get unique chords
chord_to_idx = {chord: i+1 for i, chord in enumerate(unique_chords)}

idx_to_chord = {i: chord for chord, i in chord_to_idx.items()}

for i in range(len(chord_names_train)):
    # Encode each chord in the array separately
    chord_names_train[i] = np.array([chord_to_idx[chord] for chord in chord_names_train[i]])

for i in range(len(chord_names_test)):
    # Encode each chord in the array separately
    chord_names_test[i] = np.array([chord_to_idx[chord] for chord in chord_names_test[i]])

# Add the missing chord names to the testing data with their corresponding class indices

print(chord_to_idx)
# Create PyTorch tensor

# Convert strings to numerical values or code
import torch.nn as nn

import torch
from torch.nn.utils.rnn import pad_sequence, pack_sequence

# Convert the lists of lists to lists of torch tensors
chord_durations_train = [torch.tensor(duration) for duration in chord_durations_train]
chord_durations_test = [torch.tensor(duration) for duration in chord_durations_test]

chord_names_train = [torch.tensor(chord) for chord in chord_names_train]
chord_names_test = [torch.tensor(chord) for chord in chord_names_test]


# Pad the sequences in all_chords and all_durations
chord_names_train = pad_sequence(chord_names_train, batch_first=False, padding_value=0)
chord_durations_train = pad_sequence(chord_durations_train, batch_first=False, padding_value=0)
chord_names_test = pad_sequence(chord_names_test, batch_first=False, padding_value=0)
# Swap the dimensions of all_durations_tensor so that the sequence length dimension comes first
chord_durations_test= pad_sequence(chord_durations_train, batch_first=False, padding_value=0)

# Create packed sequences from the padded sequences
chord_names_train_tensor = pack_sequence(chord_names_train, enforce_sorted=False)
chord_durations_train_tensor = pack_sequence(chord_durations_train, enforce_sorted=False)
chord_names_test_tensor = pack_sequence(chord_names_test, enforce_sorted=False)
chord_durations_test = pack_sequence(chord_durations_train, enforce_sorted=False)

chord_names_test, _ = torch.nn.utils.rnn.pad_packed_sequence(chord_names_test_tensor)

spectrograms_tensor = torch.tensor(spectrograms_train, dtype=torch.float32)
spectrograms_tensor_test = torch.tensor(spectrograms_test, dtype=torch.float32)


class ChordDataset(torch.utils.data.Dataset):
    def __init__(self, spectrograms, chord_names, chord_durations):
        self.spectrograms = spectrograms
        self.chord_names = chord_names
        self.chord_durations = chord_durations

    def __getitem__(self, index):
        spectrogram = self.spectrograms[index]
        chord_name = self.chord_names[index]
        chord_duration = self.chord_durations[index]
        return spectrogram, chord_name, chord_duration

    def __len__(self):
        return len(self.spectrograms)


all_chords_tensor, _ = torch.nn.utils.rnn.pad_packed_sequence(chord_names_train_tensor)
all_durations_tensor, _ = torch.nn.utils.rnn.pad_packed_sequence(chord_durations_train_tensor)

chord_names_test, _ = torch.nn.utils.rnn.pad_packed_sequence(chord_names_test_tensor)
chord_durations_test, _ = torch.nn.utils.rnn.pad_packed_sequence(chord_durations_test)
all_durations_tensor = all_durations_tensor[:, :4]



# Create a tensor of zeros with size [32, 97]




train_dataset = ChordDataset(spectrograms_tensor, all_chords_tensor, all_durations_tensor)
test_dataset = ChordDataset(spectrograms_test, chord_names_test, chord_durations_test)

import torch
from torch.nn.utils.rnn import pad_sequence

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch.nn.functional as F


# The model
class CNNLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(CNNLSTMModel, self).__init__()
        # Define CNN layers
        self.conv1 = nn.Conv2d(in_channels=600, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        # Define LSTM layer
        self.lstm = nn.LSTM(input_size=4000, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        # Define output layers for chords and durations
        self.fc_chords = nn.Linear(in_features=hidden_size, out_features=num_classes)
        self.fc_durations = nn.Linear(in_features=hidden_size, out_features=4)

    def forward(self, x):
        # Pass input through CNN layers
        print("BEGIN")
        print(x.shape)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # Reshape the output for LSTM
        x = x.contiguous().view(x.size(0), x.size(1), -1)
        # Pass input through LSTM layer
        x, _ = self.lstm(x)

        # Flatten LSTM output
        x = x.contiguous().view(-1, x.size(-1))
        # Pass LSTM output through output layers for chords and durations
        chords_output = self.fc_chords(x)
        durations_output = self.fc_durations(x)
        seq_len = int(chords_output.size(0) / 32)
        chords_output = chords_output.view(32, seq_len, chords_output.size(1))


        return chords_output, durations_output


batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

input_size = spectrograms_train.shape[3]
hidden_size = 256
num_layers = 2

num_classes = 17
model = CNNLSTMModel(input_size, hidden_size, num_layers, num_classes)

# Define loss function


criterion_chords = nn.CrossEntropyLoss(ignore_index=0)


def decode_chords(chords_output):
    result = []

    for chord in chords_output:
        element = np.argmax(chord.detach().numpy())
        for key, value in chord_to_idx.items():
            if value == element:
                final = key
                result.append(final)
                break
    return result

def decode_chord_key(Chords):
    result = []
    for chord in Chords:
        if(chord!=0.):
            for key, value in chord_to_idx.items():
                if value == chord:
                    final = key
                    result.append(final)
                    break
    return result

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=.8)

# Train the model
num_epochs = 12
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (spectrogram, chord_name, chord_duration) in enumerate(train_loader):
        if chord_name.size(0) != 32:
            continue

        chords_output, durations_output = model(spectrogram)


        # Compute loss, ignoring padded val




        pad_size = chord_name.size(1) - chords_output.size(1)
        chords_output = F.pad(chords_output, (0, 0, 0, pad_size), "constant", 0)
        chords_output = chords_output.view(-1,num_classes)
        chord_name = chord_name.view(-1)


        loss1 = criterion_chords(chords_output, chord_name)

        durations_output = durations_output.float()
        chord_duration = chord_duration.float()

        # Compute chords loss
        # durations_output = durations_output[:, 0,:]

        # loss2 = criterion_duration(durations_output, chord_duration)
        # print(loss2.item())
        #print("END OF LOSS")
        loss = loss1
        loss.backward()
        optimizer.step()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item()}')




