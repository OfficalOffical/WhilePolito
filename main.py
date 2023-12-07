import librosa
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

dev_data = []
eval_data = []

def read_audio(file_path):
    # Load an audio file as a floating point time series.
    audio, sample_rate = librosa.load(file_path, sr=None)
    return audio, sample_rate



dev_folder = 'dev'  # Replace with the path to your dev folder


# List all .wav files in the dev folder
dev_files = [os.path.join(dev_folder, file) for file in os.listdir(dev_folder) if file.endswith('.wav')]
eval_files = [os.path.join(dev_folder, file) for file in os.listdir(dev_folder) if file.endswith('.wav')]


def reduce_noise_by_threshold(audio, threshold):
    # Apply thresholding to the audio signal
    reduced_noise_audio = np.where(np.abs(audio) < threshold, 0, audio)
    return reduced_noise_audio

# Iterate over the list of files and read each file
for file in dev_files:
    audio, sample_rate = read_audio(file)
    audio = reduce_noise_by_threshold(audio, threshold=0.005)
    dev_data.append(audio)

for file in eval_files:
    audio, sample_rate = read_audio(file)
    audio = reduce_noise_by_threshold(audio, threshold=0.005)
    eval_data.append(audio)




def extract_label(file_path):
    file_path = file_path[-5:-4]
    return file_path




dev_data = np.array(dev_data, dtype=object)
eval_data = np.array(eval_data, dtype=object)


#create labels using dev files and extract label
dev_labels = [extract_label(file) for file in dev_files]
eval_labels = [extract_label(file) for file in eval_files]





class SimpleCNN1D(nn.Module):
    def __init__(self):
        super(SimpleCNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1)

        # Dummy input for size calculation
        self._to_linear = None
        self._calculate_to_linear(1500)  # Assuming 1500 is the initial sequence length

        self.fc1 = nn.Linear(self._to_linear, 64)
        self.fc2 = nn.Linear(64, 10)  # Assuming 10 classes for classification

    def _calculate_to_linear(self, sequence_length):
        # Dummy input for determining size after conv layers
        dummy_data = torch.zeros((1, 1, sequence_length))
        dummy_data = self.pool(F.relu(self.conv1(dummy_data)))
        dummy_data = self.pool(F.relu(self.conv2(dummy_data)))
        self._to_linear = int(np.prod(dummy_data.size()))

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self._to_linear)  # Flatten for linear layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Example data
X_train = torch.randn(100, 1, 1500)  # 100 samples, 1 channel, 1500 data points
y_train = torch.randint(0, 10, (100,))  # 100 labels, for instance, in a classification task with 10 classes

# Create dataset and data loader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

model = SimpleCNN1D()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Example Training Loop with Accuracy Calculation
for epoch in range(10):  # Number of epochs
    total = 0
    correct = 0

    for data, target in train_loader:
        optimizer.zero_grad()  # Zero the gradient buffers
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    train_accuracy = 100 * correct / total
    print(f'Epoch {epoch+1}, Loss: {loss.item()}, Accuracy: {train_accuracy}%')





model.eval()  # Set model to evaluation mode

# Assuming you have a validation loader: val_loader
val_total = 0
val_correct = 0


eval_data_tensor = torch.tensor(eval_data, dtype=torch.float32)
eval_labels_tensor = torch.tensor(eval_labels, dtype=torch.long)
eval_dataset = TensorDataset(eval_data_tensor, eval_labels_tensor)
val_loader = DataLoader(eval_dataset, batch_size=32, shuffle=True)




with torch.no_grad():  # Disable gradient calculation for validation

    for data, target in val_loader:
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        val_total += target.size(0)
        val_correct += (predicted == target).sum().item()

        print(predicted, target)

    val_accuracy = 100 * val_correct / val_total
    print(f'Validation Accuracy: {val_accuracy}%')




