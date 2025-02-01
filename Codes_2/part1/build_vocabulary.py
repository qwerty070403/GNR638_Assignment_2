import cv2
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
from time import time
import os

def build_vocabulary(image_paths, vocab_size):
    """
    Input:
        image_paths: List of training image paths
        vocab_size: Number of clusters desired
    Output:
        vocab: Cluster centers from K-means
    """
    sift = cv2.SIFT_create()  # Initialize SIFT detector
    bag_of_features = []

    print("Extracting SIFT features...")
    for path in image_paths:
        # Load image in grayscale
        img = Image.open(path).convert('L')
        img = np.array(img)

        # Detect and compute SIFT features
        keypoints, descriptors = sift.detectAndCompute(img, None)
        if descriptors is not None:
            bag_of_features.append(descriptors)

    # Combine all descriptors into a single array
    bag_of_features = np.vstack(bag_of_features).astype('float32')

    print("Performing K-means clustering...")
    start_time = time()
    kmeans = KMeans(n_clusters=vocab_size, init='k-means++', random_state=42)
    kmeans.fit(bag_of_features)
    vocab = kmeans.cluster_centers_
    end_time = time()

    print(f"Vocabulary computed in {end_time - start_time:.2f} seconds.")
    return vocab
