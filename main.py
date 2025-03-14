import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from PIL import Image
from torchvision.transforms import v2

# Dataset Class with Image Limit
class WasteDataset(Dataset):
    def __init__(self, folder_path, transform=None, max_images=None):
        self.folder_path = folder_path
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_names = sorted(os.listdir(folder_path))

        for label, class_name in enumerate(self.class_names):
            class_path = os.path.join(folder_path, class_name)
            image_files = [file for file in os.listdir(class_path) if file.endswith('.jpg')]        

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
    v2.ElasticTransform(alpha=1, sigma=0.2),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    v2.RandomHorizontalFlip(p=0.5)
])

# Dataset Paths
dataset_path = "wastes"

# Load Dataset 
full_dataset = WasteDataset(os.path.join(dataset_path, 'train'), transform=transform)
test_dataset = WasteDataset(os.path.join(dataset_path, 'test'), transform=transform)

# Split into train (80%) and validation (20%)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, pin_memory=True)

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
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
    

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # Adaptive pooling layer
        self.flatten = nn.Flatten()

        self.dropout1 = nn.Dropout(p=0.3)  # Increased dropout

        self.linear1 = nn.Linear(256, 512)
        self.linear2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)

        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)

        x = self.global_avg_pool(x) # The pooling becomes adaptive

        x = self.flatten(x)
        x = self.relu(self.linear1(x))
        x = self.dropout1(x)
        x = self.linear2(x)

        return x


# Training Function 
def train_model(model, train_loader, val_loader, loss_fn, optimizer, num_epochs=1, save_path="waste_classification.pth"):
    if os.path.dirname(save_path):  # Only create if a directory is specified
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
    model.to(device)
    best_val_acc = 0.0  # Track best validation accuracy
    losses = []  # Track training loss history
    val_losses = []  # Track validation loss history

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for train_inputs, train_outputs in train_loader:
            train_inputs, train_outputs = train_inputs.to(device), train_outputs.to(device)  # Move data to GPU
            optimizer.zero_grad()  # Reset gradients

            pred = model(train_inputs)  # Forward pass

            train_outputs = train_outputs.long()  # Ensure labels are long for CrossEntropyLoss
            loss = loss_fn(pred, train_outputs)  # Compute loss

            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)  # Compute average loss
        losses.append(avg_train_loss)
        
        # Run the validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for val_inputs, val_labels in val_loader:
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                val_outputs = model(val_inputs)

                loss = loss_fn(val_outputs, val_labels.long())
                val_loss += loss.item()

                _, predicted = torch.max(val_outputs, 1)
                total += val_labels.size(0)
                correct += (predicted == val_labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_acc = 100 * correct / total

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")


        #save the model
        #model_save_path = save_path.replace(".pth", f"_epoch_{epoch+1}.pth")
        #torch.save(model.state_dict(), model_save_path)
        #print(f"Model saved")

    return model, losses, val_losses

def test_model(model, test_loader):
    model.to(device)
    model.eval()
    correct = 0 
    total = 0 

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print (f"Test Accuracy: {accuracy:.2f}%")
    return accuracy        

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Initialize Model
model = NeuralNet().to(device)

# Training Configuration
NUM_EPOCHS = 5

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.01)

# Train Model
trained_model, train_loss, val_loss = train_model(model, train_loader, val_loader, loss_fn, optimizer, num_epochs=NUM_EPOCHS, save_path="models/waste_classification.pth")

# Test Model
test_accuracy = test_model(trained_model, test_loader)

# Print Loss History
# print("Loss History:", loss_history)
