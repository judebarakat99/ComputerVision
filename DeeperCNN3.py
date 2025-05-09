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
import random

#####################################################
# ---------- Step 1: Load the Dataset ----------
#####################################################

DATASET_PATH = "rgbd-dataset"  # Ensure the correct path is used

categories = [cat for cat in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, cat))]
categories = categories[:20]  # Use the first 20 categories, this is the same as the CV code
print("Classes found:", categories)

# Function to load images from a specific category
# This function loads images from the dataset, resizes them, and returns a list of images
# I am using this sample function to load images from the dataset in the CV code
# I was getting errors when I was using the original code, so I changed it to this muliple nested if statements even though it is not the cleanest code it works fine.
# this is loading from different viewpoints, so this related to the better use for real life applicaions using robotics, since robots are always moving from dofferent viewpoints
def load_images(category, max_imgs=100):
    images = []
    cat_path = os.path.join(DATASET_PATH, category)
    for viewpoint_folder in os.listdir(cat_path):
        viewpoint_path = os.path.join(cat_path, viewpoint_folder)
        if os.path.isdir(viewpoint_path):
            for img_name in os.listdir(viewpoint_path)[:max_imgs]:
                if img_name.endswith(".png"):
                    img_path = os.path.join(viewpoint_path, img_name)
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.resize(img, (128, 128))
                        images.append(img)
    return images

# Load images and labels
data = []
labels = []
for category in categories:
    imgs = load_images(category, max_imgs=100)
    data.extend(imgs)
    labels.extend([category] * len(imgs))

data = np.array(data) / 255.0  # Normalize
labels = np.array(labels)

print("Total images loaded:", len(data))

# Encode labels
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
labels_categorical = tf.keras.utils.to_categorical(labels_encoded, num_classes=len(categories))

# Train-test split
# Split the dataset into training and testing sets
# I am using this sample function to split the dataset into training and testing sets
# as the tradition most commonly we split the data to 80% of the data is used for training and 20% is used for testing.
X_train, X_test, y_train, y_test = train_test_split(data, labels_categorical, test_size=0.2, random_state=42)

#####################################################
# ---------- Step 2: Define the CNN Model ----------
#####################################################
# Define the CNN model
# I am using this sample function to define the CNN model
# this function defines a deep CNN model with several convolutional and pooling layers
model = models.Sequential([
    # the original code I inspires this from was using input_shape, but I kept getting errors, so I chsnged it to shape
    layers.InputLayer(shape=(128, 128, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.5),  # Regularization to reduce overfitting
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(categories), activation='softmax')
])

# Compile model
# we learned from machine learning course the types of optimizers and loss functions
# I am using this sample function to compile the model, it was easiest one to use
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Early stopping
# I am using this sample function to stop the training if the validation loss does not improve for 3 epochs, to not waste time and resources and to prevent overfitting
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

#####################################################
# ---------- Step 3: Train the CNN Model ----------
#####################################################
print("Training the model...")
history = model.fit(
    X_train, y_train,
    epochs=10, # I can increase this number for better results, but it will take longer and for the purooses of this report I am using A LOW NUMBER for speed and it was already enough to get the results needed for this report
    batch_size=32,
    validation_data=(X_test,y_test),
    callbacks=[early_stop]
)

#####################################################
# ---------- Step 4: Test the Model ----------
#####################################################
# Evaluate the model on the test set
print("\nEvaluating the model on the test set...")
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy * 100:.2f}%")


#####################################################
# ---------- Step 5: Plot Training/Validation History ----------
#####################################################


# I am using this function to plot the training and validation accuracy and loss, they are in the same plot to :
# 1. save room on the report :) 
# 2. in the codes I saw in resources, they used this as well
# 3. it is easier to see the difference between the training and validation accuracy and loss
os.makedirs("graphs_big_dataset", exist_ok=True)

# Accuracy plot
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig("graphs_big_dataset/cnn_accuracy_plot.png")

# Loss plot
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig("graphs_big_dataset/cnn_loss_plot.png")

#####################################################
# ---------- Step 6: Save Results ----------
#####################################################
# I am also using this code in the cv code to save the results
# Create the directory to save results if it doesn't exist
os.makedirs("results_big_dataset", exist_ok=True)

# Predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# Convert to class names
y_pred_names = le.inverse_transform(y_pred_classes)
y_test_names = le.inverse_transform(y_test_classes)

# Classification report and confusion matrix
report = classification_report(y_test_names, y_pred_names, zero_division=1, target_names=categories)
matrix = confusion_matrix(y_test_names,y_pred_names)

with open("results_big_dataset/deep_CNN_results.txt", "w") as f:
    f.write("Deep CNN Report:\n")
    f.write(report+"\n\n")
    f.write("Confusion Matrix:\n")
    for row in matrix:
        f.write(" ".join(map(str, row)) + "\n")

print("Deep CNN results saved to deep_CNN_results.txt")
