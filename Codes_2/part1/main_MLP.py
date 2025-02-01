#Note this code uses 3-Layer MLP classifier

import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from google.colab import drive
from build_vocabulary import build_vocabulary
from get_bags_of_sifts import get_bags_of_sifts
from PIL import Image
from sklearn.cluster import KMeans
import numpy as np
from scipy.spatial import distance
import pickle
from time import time
import cv2


# Set the folder path to the dataset directory
folder_path = "/content/drive/MyDrive/Images"  # Update as necessary

# Check if the folder exists
if not os.path.exists(folder_path):
    raise ValueError(f"The folder path '{folder_path}' does not exist. Please check the path.")

# Get all image files recursively from subfolders
valid_extensions = ('.tif', '.tiff', '.png', '.jpg', '.jpeg')  # Add other extensions if needed
all_images = []
for root, _, files in os.walk(folder_path):
    for file in files:
        if file.lower().endswith(valid_extensions):
            all_images.append(os.path.join(root, file))

# Ensure images were found
if len(all_images) == 0:
    raise ValueError(f"No images found in the folder '{folder_path}'. Please check the folder structure and file extensions.")

# Shuffle and split the dataset
random.shuffle(all_images)

# Split into train, validation, and test
train_images, temp_images = train_test_split(all_images, test_size=0.3, random_state=42)
val_images, test_images = train_test_split(temp_images, test_size=2/3, random_state=42)

print(f"Training images: {len(train_images)}")
print(f"Validation images: {len(val_images)}")
print(f"Testing images: {len(test_images)}")

#dataset splitted

# Number of codewors in clusters
num_clusters = 500 # Vary number of codewords accordingly, 500 value gives best train and validation accuracy 

# Build vocabulary
vocab = build_vocabulary(train_images, vocab_size=num_clusters)

# Save the vocabulary
with open('vocab.pkl', 'wb') as f:
    pickle.dump(vocab, f)

print("Vocabulary created and saved.")



# Extract features for training and validation sets
train_features = get_bags_of_sifts(train_images)
val_features = get_bags_of_sifts(val_images)

# Assign labels to training and validation images (e.g., based on folder structure)
train_labels = [os.path.basename(os.path.dirname(img)) for img in train_images]
val_labels = [os.path.basename(os.path.dirname(img)) for img in val_images]

print("Bags of SIFT features extracted for train and validation sets.")



# Define an experiment function
def run_experiment(hidden_dim1, hidden_dim2, activation_fn, save_path):
    class MLP(nn.Module):
        def __init__(self, input_dim, output_dim=21):  # Assuming 21 classes in UCM
            super(MLP, self).__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim1)
            self.activation1 = activation_fn
            self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
            self.activation2 = activation_fn
            self.fc3 = nn.Linear(hidden_dim2, output_dim)

        def forward(self, x):
            x = self.activation1(self.fc1(x))
            x = self.activation2(self.fc2(x))
            x = self.fc3(x)
            return x

    # Ensure labels are encoded as integers
    label_encoder = LabelEncoder()
    train_labels_encoded = label_encoder.fit_transform(train_labels)
    val_labels_encoded = label_encoder.transform(val_labels)

    # Convert data to tensors
    train_X = torch.tensor(train_features, dtype=torch.float32)
    val_X = torch.tensor(val_features, dtype=torch.float32)
    train_y = torch.tensor(train_labels_encoded, dtype=torch.long)
    val_y = torch.tensor(val_labels_encoded, dtype=torch.long)

    input_dim = train_features.shape[1]  # Number of features per image
    model = MLP(input_dim=input_dim)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 1000
    best_val_acc = 0.0  # Track best validation accuracy

    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(train_X)
        loss = criterion(outputs, train_y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validation accuracy
        with torch.no_grad():
            train_preds = torch.argmax(model(train_X), dim=1)
            val_preds = torch.argmax(model(val_X), dim=1)
            train_acc = (train_preds == train_y).float().mean().item() * 100  # Convert to percentage
            val_acc = (val_preds == val_y).float().mean().item() * 100  # Convert to percentage

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss.item():.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

        # Save the best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"Model saved at {save_path} with val accuracy: {val_acc:.2f}%")

    print("Training completed. Best model saved at:", save_path)
    return save_path  # Return the path of the saved model



#Training Phase 


#choose best model vary hidden layer size and activation function (nn.Tanh,nn.ReLU,nn.LeakyReLU(0.01))
saved_model_path = run_experiment(hidden_dim1=512, hidden_dim2=256, activation_fn=nn.Tanh(), save_path="mlp_model_512_256_tanh")
print(f"Model successfully saved at {saved_model_path}")



#Testing Phase


# Extract features for test
test_features = get_bags_of_sifts(test_images)

# Assign labels to testing
test_labels = [os.path.basename(os.path.dirname(img)) for img in test_images]
# Ensure labels are encoded as integers
label_encoder = LabelEncoder()
test_labels_encoded = label_encoder.fit_transform(test_labels)

print("Bags of SIFT features extracted for test sets.")



class MLP(nn.Module):
        def __init__(self, input_dim, output_dim=21):  # Assuming 21 classes in UCM
            super(MLP, self).__init__()
            self.fc1 = nn.Linear(input_dim, 512)
            self.activation1 = nn.Tanh()
            self.fc2 = nn.Linear(512,256)
            self.activation2 = nn.Tanh()
            self.fc3 = nn.Linear(256, output_dim)

        def forward(self, x):
            x = self.activation1(self.fc1(x))
            x = self.activation2(self.fc2(x))
            x = self.fc3(x)
            return x


test_X = torch.tensor(test_features, dtype=torch.float32)
test_y = torch.tensor(test_labels_encoded, dtype=torch.long)

# Load the best saved model
model = MLP(input_dim=test_features.shape[1])
model.load_state_dict(torch.load(saved_model_path))
model.eval()  # Set to evaluation mode

# Compute test accuracy
with torch.no_grad():
    test_preds = torch.argmax(model(test_X), dim=1)
    test_acc = (test_preds == test_y).float().mean().item()

print(f" Test Accuracy: {test_acc*100:.4f}%")
