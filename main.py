import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from torchvision.transforms import v2

# Dataset Class with Image Limit
class WasteDataset(Dataset):
    def __init__(self, folder_path, transform=None, max_images=100):
        self.folder_path = folder_path
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_names = sorted(os.listdir(folder_path))

        for label, class_name in enumerate(self.class_names):
            class_path = os.path.join(folder_path, class_name)
            image_files = [file for file in os.listdir(class_path) if file.endswith('.jpg')]

            # Limit the number of images per class to `max_images`
            image_files = image_files[:max_images]

            for file in image_files:
                self.images.append(os.path.join(class_path, file))
                self.labels.append(label)  # Numeric label for each class

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")  # Open image
        
        if self.transform:
            image = self.transform(image)  # Apply transformations
        
        return image, torch.tensor(label, dtype=torch.long)  # Convert label to long


# Define Transformations
transform = v2.Compose([
    v2.ToTensor(),
    v2.Resize((224, 224)),
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.3, hue=0.1),
    v2.RandomRotation(degrees=180),
    v2.GaussianNoise(mean=0.0, sigma=0.1),
    v2.ElasticTransform(alpha=1, sigma=0.2)
])

# Dataset Paths
dataset_path = "/home/CMPM17-ML/CMPM-17-Final-WasteClassification/wastes"

# Load Dataset with Image Cap
train_dataset = WasteDataset(os.path.join(dataset_path, 'train'), transform=transform, max_images=100)

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Print Training Data Information
print("Training Data:")
for batch_idx, (inputs, targets) in enumerate(train_loader):
    print(f"Batch {batch_idx + 1}:")
    print(f"Input (image tensor shape): {inputs.shape}")
    print(f"Output (labels): {targets}")

# Define Neural Network Model with Dropout Fix
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()

        # Dropout layers
        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.5)

        # Fully connected layers
        self.linear1 = nn.Linear(64 * 56 * 56, 1028)
        self.linear2 = nn.Linear(1028, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)

        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.pool(x)

        x = self.flatten(x)
        x = self.relu(self.linear1(x))
        x = self.dropout1(x)

        x = self.dropout2(x)
        x = self.linear2(x)

        return x


# Training Function with Fixed Model Saving
def train_model(model, train_loader, loss_fn, optimizer, num_epochs=1, save_path="waste_classification.pth"):
    if os.path.dirname(save_path):  # Only create if a directory is specified
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    model.train()  # Set model to training mode
    losses = []  # Track loss for each epoch

    for epoch in range(num_epochs):
        running_loss = 0.0

        for train_inputs, train_outputs in train_loader:
            optimizer.zero_grad()  # Reset gradients

            pred = model(train_inputs)  # Forward pass

            train_outputs = train_outputs.long()  # Ensure labels are long for CrossEntropyLoss
            loss = loss_fn(pred, train_outputs)  # Compute loss

            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)  # Compute average loss
        losses.append(avg_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        #save the model
        model_save_path = save_path.replace(".pth", f"_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved")

    print("Training Complete!")
    return model, losses


# Initialize Model
model = NeuralNet()

# Training Configuration
NUM_EPOCHS = 3
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

# Train Model
trained_model, loss_history = train_model(model, train_loader, loss_fn, optimizer, num_epochs=NUM_EPOCHS, save_path="models/waste_classification.pth")

# Print Loss History
print("Loss History:", loss_history)

