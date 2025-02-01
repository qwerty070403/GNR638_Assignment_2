from PIL import Image
import numpy as np
from scipy.spatial import distance
import pickle
from time import time
import cv2
import os

def get_bags_of_sifts(image_paths):
    """
    Input:
        image_paths: List of image paths
    Output:
        image_feats: (N, d) feature matrix, where each row represents a feature vector of an image
    """
    # Load the precomputed vocabulary (k-means cluster centers)
    with open('vocab.pkl', 'rb') as handle:
        vocab = pickle.load(handle)

    vocab = np.array(vocab)  # Ensure vocab is a NumPy array
    vocab_size = vocab.shape[0]
    sift = cv2.SIFT_create()  # Initialize SIFT detector

    image_feats = []

    start_time = time()
    print("Constructing bags of SIFTs...")

    for path in image_paths:
        # Load the image in grayscale
        img = Image.open(path).convert('L')
        img = np.array(img)

        # Extract SIFT features
        keypoints, descriptors = sift.detectAndCompute(img, None)
        if descriptors is None:
            # Handle images with no detected SIFT features
            descriptors = np.zeros((1, 128), dtype='float32')

        # Compute the distance from each descriptor to each cluster center
        dist = distance.cdist(descriptors, vocab, metric='euclidean')

        # Assign each descriptor to the nearest cluster center
        idx = np.argmin(dist, axis=1)

        # Create a histogram of visual word occurrences
        hist, _ = np.histogram(idx, bins=np.arange(vocab_size + 1))

        # Normalize the histogram
        hist_norm = hist.astype('float32') / np.sum(hist)

        # Append the histogram to the feature list
        image_feats.append(hist_norm)

    image_feats = np.array(image_feats)

    end_time = time()
    print(f"It took {end_time - start_time:.2f} seconds to construct bags of SIFTs.")

    return image_feats
