import os
import numpy as np
from PIL import Image

#Preprocessing Images

# Image resizing function
img_size = 72  # Target size (72x72)

def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")  # Ensure 3 channels (RGB)
    img = img.resize((img_size, img_size))  # Resize
    img = np.array(img) / 255.0  # Normalize pixel values (0-1)
    return img.flatten()  # Flatten to 1D

# Folder path containing images
folder_path = "/content/drive/MyDrive/Images"

# Get all image file paths
all_images = [os.path.join(root, file) for root, _, files in os.walk(folder_path) 
              for file in files if file.lower().endswith(('tif', 'tiff', 'png', 'jpg', 'jpeg'))]

# Ensure images exist
if not all_images:
    raise ValueError("No images found in the dataset folder!")

print(f"Total images found: {len(all_images)}")


#Linearize Images

# Convert images to 1D vectors with progress tracking
image_data = []
total_images = len(all_images)

for idx, img_path in enumerate(all_images, start=1):
    image_data.append(preprocess_image(img_path))
    
    # Print progress every 100 images (you can adjust this)
    if idx % 100 == 0 or idx == total_images:
        print(f"Processed {idx}/{total_images} images...")

# Convert to NumPy array
X = np.array(image_data)  # Feature matrix
print(f"Shape of dataset: {X.shape}")  # Should be (num_images, 72*72*3)


# 3-Layered MLP Classifier

# Define the experiment function
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

    num_epochs = 550
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


saved_model_path = run_experiment(hidden_dim1=512, hidden_dim2=256, activation_fn=nn.LeakyReLU(0.01), save_path="mlp_model_512_256_leaky_RelU")
print(f"Model successfully saved at {saved_model_path}")


#Testing Phase




# Ensure labels are encoded consistently across both train and test data
label_encoder = LabelEncoder()

# Fit the label encoder on the entire set of labels (train + test)
all_labels = np.concatenate([train_labels, test_labels])
label_encoder.fit(all_labels)

# Now, transform both the train and test labels
train_labels_encoded = label_encoder.transform(train_labels)
val_labels_encoded = label_encoder.transform(val_labels)
test_labels_encoded = label_encoder.transform(test_labels)

# Convert data to tensors
train_X = torch.tensor(train_features, dtype=torch.float32)
val_X = torch.tensor(val_features, dtype=torch.float32)
test_X = torch.tensor(test_features, dtype=torch.float32)

train_y = torch.tensor(train_labels_encoded, dtype=torch.long)
val_y = torch.tensor(val_labels_encoded, dtype=torch.long)
test_y = torch.tensor(test_labels_encoded, dtype=torch.long)

# Define the MLP class (already provided in your code)
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim=21):  # Assuming 21 classes in UCM
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.activation1 = nn.LeakyReLU(0.01)
        self.fc2 = nn.Linear(512, 256)
        self.activation2 = nn.LeakyReLU(0.01)
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = self.activation1(self.fc1(x))
        x = self.activation2(self.fc2(x))
        x = self.fc3(x)
        return x

# Load the saved model
input_dim = test_features.shape[1]  # Assuming your features are the same size across train/test
model = MLP(input_dim=input_dim)
model.load_state_dict(torch.load("/content/mlp_model_512_256_leaky_RelU"))
model.eval()

# Evaluate on test set
with torch.no_grad():
    test_preds = torch.argmax(model(test_X), dim=1)
    test_acc = (test_preds == test_y).float().mean().item() * 100  # Convert to percentage

print(f"Test Accuracy: {test_acc:.2f}%")

