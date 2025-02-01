#Note this code uses SVM classifier

import os
import random
from sklearn.model_selection import train_test_split
from google.colab import drive
from build_vocabulary import build_vocabulary
from get_bags_of_sifts import get_bags_of_sifts
from svm_classify import svm_classify
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
num_clusters = 500 # Vary number of codewords accordingly, 500 value gives best train and validation accuracy when SVM classifier is used

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


# Predict labels for validation images
val_predictions = svm_classify(train_features, train_labels, val_features)

# Compute train accuracy
train_predictions = svm_classify(train_features, train_labels, train_features)
train_accuracy = sum([pred == actual for pred, actual in zip(train_predictions, train_labels)]) / len(train_labels)

# Compute validation accuracy
val_accuracy = sum([pred == actual for pred, actual in zip(val_predictions, val_labels)]) / len(val_labels)

print(f"Train Accuracy: {train_accuracy * 100:.2f}%")
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

#Here we get Train and Validation accuracy
#Comment out the entire part below and run till we get best train and validation accuracy
#Once we get the best train and validation accuracy Uncomment the part below and run to get the test accuracy



# Extract features for the test set
test_features = get_bags_of_sifts(test_images)

# Assign labels to test images (same as train/validation labeling method)
test_labels = [os.path.basename(os.path.dirname(img)) for img in test_images]

print("Bags of SIFT features extracted for the test set.")


# Predict labels for test images
test_predictions = nearest_neighbor_classify(train_features, train_labels, test_features)
# Compute test accuracy
test_accuracy = sum([pred == actual for pred, actual in zip(test_predictions, test_labels)]) / len(test_labels)

print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
