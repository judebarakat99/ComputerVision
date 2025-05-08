import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
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
#                     img = cv2.resize(img, (128, 128))
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


# Load a subset of images from each category (100 images per class, 20 classes)
data = []
labels = []

for category in categories:
    imgs = load_images(category, max_imgs=100)
    data.extend(imgs)
    labels.extend([category] * len(imgs))

print("Total images loaded:", len(data))

# ---------- Step 2: Extract SIFT Features ----------
sift = cv2.SIFT_create()
descriptor_list = []
image_descriptors = []

print("Extracting SIFT features...")

for img in tqdm(data):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    if descriptors is not None:
        descriptor_list.extend(descriptors)
        image_descriptors.append(descriptors)
    else:
        image_descriptors.append(np.array([]))  # Placeholder

# ---------- Step 3: Create Visual Vocabulary (Bag of Words) ----------
K = 100  # Number of visual words
descriptor_array = np.vstack([d for d in image_descriptors if d.size > 0])

print("Building visual vocabulary with KMeans...")
kmeans = KMeans(n_clusters=K, random_state=42)
kmeans.fit(descriptor_array)

# Build histogram for each image
def build_histogram(descriptors, kmeans, K):
    if descriptors.size == 0:
        return np.zeros(K)
    cluster_indices = kmeans.predict(descriptors)
    histogram = np.zeros(K)
    for idx in cluster_indices:
        histogram[idx] += 1
    return histogram

features = np.array([build_histogram(desc, kmeans, K) for desc in image_descriptors])

# ---------- Step 4: Train and Evaluate SVM Classifier ----------
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ---------- Step 5: Save Results ----------
# Ensure the results folder exists
os.makedirs("results_big_dataset", exist_ok=True)

# Save classification report and confusion matrix to a file
with open("results_big_dataset/traditional_CV_results.txt", "w") as f:
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)

    f.write("Traditional CV Report:\n")
    f.write(report + "\n\n")
    
    f.write("Confusion Matrix:\n")
    for row in matrix:
        f.write(" ".join(map(str, row)) + "\n")

print("✅ SIFT+SVM results saved to traditional_CV_results.txt")
