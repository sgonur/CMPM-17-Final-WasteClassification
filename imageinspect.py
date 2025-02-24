import os
import matplotlib.pyplot as plt
from PIL import Image
import random

# Function to load images and labels from a folder
def load_images_and_labels(folder_path, num_images_per_class=10):
    images = []
    labels = []
    
    # Iterate through each class folder
    for class_name in os.listdir(folder_path):
        class_path = os.path.join(folder_path, class_name)
        
        # Gets the image paths from each folder
        image_files = []
        for f in os.listdir(class_path):
            if f.endswith('.jpg'):
                full_path = os.path.join(class_path, f)
                image_files.append(full_path)
        
        random.shuffle(image_files) #shuffle the images each time so we don't get the same data everytime
        image_files = image_files[:num_images_per_class]  # Limit the number of images per class
        
        # Load images and assign labels
        for img_path in image_files:
            img = Image.open(img_path)
            images.append(img)
            labels.append(class_name)
    
    return images, labels

# Function to display images with labels and save the plot
def save_images_with_labels(images, labels, output_path, num_images=100, rows=10, cols=10):
    # This makes the size of the canvas to place the images
    plt.figure(figsize=(cols * 2, rows * 2))  # Adjust the figure size as needed

    # Loop through the images and display them
    for idx in range(min(num_images, len(images))):  # Ensure we don't exceed the number of images
        plt1 = plt.subplot(rows, cols, idx + 1)
        plt1.imshow(images[idx])
        plt1.set_title(labels[idx])
        plt1.axis('off')  # Hide the axes

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(output_path)  # Save the plot to a file
    plt.close()  # Close the figure to free up memory

# Path to the dataset
dataset_path = '/home/CMPM17-ML/CMPM-17-Final-WasteClassification/wastes'

# Load images and labels from the train folder
train_folder = os.path.join(dataset_path, 'train')
train_images, train_labels = load_images_and_labels(train_folder, num_images_per_class=10)

# Load images and labels from the test folder (optional)
test_folder = os.path.join(dataset_path, 'test')
test_images, test_labels = load_images_and_labels(test_folder, num_images_per_class=10)

# Save images from the train folder
train_output_path = 'train_images_grid.jpg'  # Path to save the train grid
print(f"Saving images from the train folder to {train_output_path}")
save_images_with_labels(train_images, train_labels, train_output_path, num_images=100, rows=10, cols=10)

# Save images from the test folder (optional)
test_output_path = 'test_images_grid.jpg'  # Path to save the test grid
print(f"Saving images from the test folder to {test_output_path}")
save_images_with_labels(test_images, test_labels, test_output_path, num_images=100, rows=10, cols=10)