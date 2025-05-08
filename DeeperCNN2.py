import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# ---------- Step 1: Load the Dataset ----------
DATASET_PATH = "rgbd-dataset"  # Ensure the correct path is used

categories = [cat for cat in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, cat))]
categories = categories[:20]  # Use the first 20 categories
print("Classes found:", categories)

# def load_images(category, max_imgs=100):
#     images = []
#     cat_path = os.path.join(DATASET_PATH, category)
#     for video_folder in os.listdir(cat_path)[:1]:  # Use only the first viewpoint
#         img_folder = os.path.join(cat_path, video_folder)
#         for img_name in os.listdir(img_folder)[:max_imgs]:
#             if img_name.endswith(".png"):
#                 img_path = os.path.join(img_folder, img_name)
#                 img = cv2.imread(img_path)
#                 if img is not None:
#                     img = cv2.resize(img, (128, 128))  # Resize for CNN
#                     images.append(img)
#     return images

def load_images(category, max_imgs=100):
    images = []  # Initialize an empty list to store images
    cat_path = os.path.join(DATASET_PATH, category)  # Path to the category folder
    
    # Loop through each viewpoint subfolder within the category (e.g., apple_1, apple_2, etc.)
    for viewpoint_folder in os.listdir(cat_path):  
        viewpoint_path = os.path.join(cat_path, viewpoint_folder)  # Path to the viewpoint folder
        
        # Check if the folder is a directory
        if os.path.isdir(viewpoint_path):
            # Loop through the images in the viewpoint folder
            for img_name in os.listdir(viewpoint_path)[:max_imgs]:  
                if img_name.endswith(".png"):  # Only process PNG images
                    img_path = os.path.join(viewpoint_path, img_name)  # Full path to the image
                    img = cv2.imread(img_path)  # Read the image using OpenCV
                    
                    if img is not None:  # If the image is loaded successfully
                        img = cv2.resize(img, (128, 128))  # Resize the image to 128x128
                        images.append(img)  # Add the image to the list
    
    return images  # Return the list of images


# Load images from each category (100 images per class, 20 categories)
data = []
labels = []

for category in categories:
    imgs = load_images(category, max_imgs=100)
    data.extend(imgs)
    labels.extend([category] * len(imgs))

# Convert to numpy arrays
data = np.array(data)
labels = np.array(labels)

print("Total images loaded:", len(data))

# Preprocess the images
data = data / 255.0  # Normalize images

# Encode labels
le = LabelEncoder()
labels = le.fit_transform(labels)
labels = tf.keras.utils.to_categorical(labels, num_classes=len(categories))

# ---------- Step 2: Define the CNN Model ----------
model = models.Sequential([
    layers.InputLayer(input_shape=(128, 128, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(categories), activation='softmax')  # Output layer for multi-class classification
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ---------- Step 3: Train the CNN Model ----------
history = model.fit(data, labels, epochs=20, batch_size=32, validation_split=0.2)

# ---------- Step 4: Test the Model ----------
# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(data, labels, verbose=0)

# Print test loss and accuracy
print(f"\nTest loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

# ---------- Step 5: Plot Training/Validation History ----------
# Plot training and validation accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Ensure the 'graphs' directory exists
os.makedirs("graphs", exist_ok=True)

# Save accuracy plot to the 'graphs' folder
plt.savefig("graphs/cnn_accuracy_plot.png")

# Plot training and validation loss
plt.figure()  # Create a new figure for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Save loss plot to the 'graphs' folder
plt.savefig("graphs/cnn_loss_plot.png")

# ---------- Step 6: Save Results ----------
# Make sure the results folder exists
os.makedirs("results_big_dataset", exist_ok=True)

# Save classification report and confusion matrix
y_pred = model.predict(data)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(labels, axis=1)

# Convert numerical predictions back to class names
y_pred_class_names = le.inverse_transform(y_pred_classes)
y_test_class_names = le.inverse_transform(y_test_classes)

# Classification report and confusion matrix
report = classification_report(y_test_class_names, y_pred_class_names)
matrix = confusion_matrix(y_test_class_names, y_pred_class_names)

with open("results_big_dataset/deep_CNN_results.txt", "w") as f:
    f.write("Deep CNN Report:\n")
    f.write(report + "\n\n")
    
    f.write("Confusion Matrix:\n")
    for row in matrix:
        f.write(" ".join(map(str, row)) + "\n")

print("Deep CNN results saved to deep_CNN_results.txt")
