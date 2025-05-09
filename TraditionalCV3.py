import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import random

#####################################################
# ---------- Step 1: Load the Dataset ----------
#####################################################

# Define the path to the dataset
DATASET_PATH = "rgbd-dataset"

# Get the categories (class names) from the dataset directory
categories = [cat for cat in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, cat))]
categories = categories[:20]  # Limit to the first 20 categories for the evaluation, this is the same as the CNN code
print("Classes found:", categories)

# Function to load images from a specific category
# This function loads images from the dataset, resizes them, and returns a list of images
# I am using this sample function to load images from the dataset in the CNN model
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

# Load the dataset by iterating through each category
data = []
labels=[]

# For each category, WE are loading corresponding images and add to dataset
for category in categories:
    imgs = load_images(category, max_imgs=100)
    data.extend(imgs)  # Add images to the dataset
    labels.extend([category] * len(imgs))  # Add the corresponding category label

#####################################################
# ---------- Step 2: Extract SIFT Features ----------
#####################################################
# Initialize the SIFT feature extractor
sift = cv2.SIFT_create()
descriptor_list = []
image_descriptors = []

print("Extracting SIFT features...")
# Extract SIFT features from each image
for img in tqdm(data):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
    keypoints, descriptors = sift.detectAndCompute(gray, None)  # Compute SIFT features
    if descriptors is not None:
        descriptor_list.extend(descriptors)
        image_descriptors.append(descriptors)
    else:
        image_descriptors.append(np.array([]))  # Handle empty descriptors

#####################################################
# ---------- Step 3: Create Bag of Words ----------
#####################################################

# Number of visual words (clusters)
K = 100
descriptor_array = np.vstack([d for d in image_descriptors if d.size > 0])

# Perform KMeans clustering to build the visual vocabulary
print("Building visual vocabulary with KMeans...")
kmeans =KMeans(n_clusters=K,random_state=42)
kmeans.fit(descriptor_array)

# Function to build histogram of visual word occurrences for each image
def build_histogram(descriptors, kmeans, K):
    if descriptors.size == 0:
        return np.zeros(K)  # Return empty histogram for images without descriptors
    cluster_indices =kmeans.predict(descriptors)  # Get cluster indices for the descriptors
    histogram = np.zeros(K)  # Initialize histogram
    for idx in cluster_indices:
        histogram[idx]+=1  # Increment the histogram for each visual word by one
    return histogram

# Build histograms for all images
features = np.array([build_histogram(desc, kmeans, K) for desc in image_descriptors])

#####################################################
# ---------- Step 4: Train and Evaluate SVM ----------
#####################################################

# this si to split the data into training and testing sets (80% training, 20% testing), this is the same as the CNN code
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train an SVM classifier with a linear kernel
svm = SVC(kernel='linear')
svm.fit(X_train,y_train)

# Make predictions on the test set
y_pred = svm.predict(X_test)

#####################################################
# ---------- Step 5: Save Results ----------
#####################################################
# i am also using this in the CNN code to save the results
# Create the directory to save results if it doesn't exist
os.makedirs("results_big_dataset", exist_ok=True)

# Generate classification report and confusion matrix
report = classification_report(y_test, y_pred)
matrix = confusion_matrix(y_test, y_pred)

# Save the evaluation results to a text file
with open("results_big_dataset/traditional_CV_results.txt", "w") as f:
    f.write("Traditional CV Report:\n")
    f.write("\n")
    f.write(report+"\n\n")
    f.write("Confusion Matrix:\n")
    for row in matrix:
        f.write(" ".join(map(str, row)) + "\n")  # Save confusion matrix

# Confirmation message for result saving
print("✅ SIFT+SVM results saved to traditional_CV_results.txt")
