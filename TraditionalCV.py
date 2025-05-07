import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ---------- Step 1: Load the Dataset ----------

DATASET_PATH = "rgbd-dataset"  # Make sure the path is correct

categories = [cat for cat in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, cat))]
print("Classes found:", categories)

def load_images(category, max_imgs=50):
    images = []
    cat_path = os.path.join(DATASET_PATH, category)
    for video_folder in os.listdir(cat_path)[:1]:  # Use only the first viewpoint
        img_folder = os.path.join(cat_path, video_folder)
        for img_name in os.listdir(img_folder)[:max_imgs]:
            if img_name.endswith(".png"):
                img_path = os.path.join(img_folder, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, (128, 128))
                    images.append(img)
    return images

# Load a subset of images from each category
data = []
labels = []

for category in categories[:5]:  # Use only 5 classes
    imgs = load_images(category, max_imgs=30)
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
os.makedirs("results", exist_ok=True)

# Save classification report and confusion matrix to a file
with open("results/traditional_CV_results.txt", "w") as f:
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)

    f.write("Traditional CV Report:\n")
    f.write(report + "\n\n")
    
    f.write("Confusion Matrix:\n")
    for row in matrix:
        f.write(" ".join(map(str, row)) + "\n")

print("✅ SIFT+SVM results saved to results/traditional_CV_results.txt")
