import os
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import math
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import v2


class WasteDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_names = sorted(os.listdir(folder_path))

        for label, class_name in enumerate(self.class_names):
            class_path = os.path.join(folder_path, class_name)
            for file in os.listdir(class_path):
                if file.endswith('.jpg'):
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
        
        return image, label
    
transform = v2.Compose([
    v2.ToTensor(),
    v2.Resize((224, 224)),
])

dataset_path = "/Users/badri/Documents/CMPM17-ML/CMPM-17-Final-WasteClassification/wastes"

# Create dataset instances
train_dataset = WasteDataset(os.path.join(dataset_path, 'train'), transform=transform)
test_dataset = WasteDataset(os.path.join(dataset_path, 'test'), transform=transform)

# Creaye the dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print("Training Data:")
for batch_idx, (inputs, targets) in enumerate(train_loader):
    print(f"Batch {batch_idx + 1}:")
    print(f"Input (image tensor shape): {inputs.shape}")
    print(f"Output (labels): {targets}")

print("\nTesting Data:")
for batch_idx, (inputs, targets) in enumerate(test_loader):
    print(f"Batch {batch_idx + 1}:")
    print(f"Input (image tensor shape): {inputs.shape}")
    print(f"Output (labels): {targets}")

