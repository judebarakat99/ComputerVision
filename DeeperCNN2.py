import os
import cv2
import numpy as np
import tensorflow as tf
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

def load_images(category, max_imgs=100):
    images = []
    cat_path = os.path.join(DATASET_PATH, category)
    for video_folder in os.listdir(cat_path)[:1]:  # Use only the first viewpoint
        img_folder = os.path.join(cat_path, video_folder)
        for img_name in os.listdir(img_folder)[:max_imgs]:
            if img_name.endswith(".png"):
                img_path = os.path.join(img_folder, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, (128, 128))  # Resize for CNN
                    images.append(img)
    return images

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
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42)

model.fit(X_train, y_train, epochs=10, batch_size=32)

# ---------- Step 4: Evaluate the Model ----------
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

# Classification report and confusion matrix
print("Deep CNN Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ---------- Step 5: Save Results ----------
# Ensure the results folder exists
os.makedirs("results", exist_ok=True)

# Save classification report and confusion matrix to a file
with open("results/deep_CNN_results.txt", "w") as f:
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)

    f.write("Deep CNN Report:\n")
    f.write(report + "\n\n")
    
    f.write("Confusion Matrix:\n")
    for row in matrix:
        f.write(" ".join(map(str, row)) + "\n")

print("✅ Deep CNN results saved to results/deep_CNN_results.txt")
