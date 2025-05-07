import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# ---------- Setup ----------
DATASET_PATH = "rgbd-dataset"
categories = [cat for cat in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, cat))]
categories = categories[:5]
label_map = {cat: i for i, cat in enumerate(categories)}
reverse_map = {v: k for k, v in label_map.items()}

# ---------- Dataset ----------
class RGBDDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

# Load image paths
image_paths = []
label_indices = []
for category in categories:
    cat_path = os.path.join(DATASET_PATH, category)
    for video_folder in os.listdir(cat_path)[:1]:
        img_folder = os.path.join(cat_path, video_folder)
        for img_name in os.listdir(img_folder)[:30]:
            if img_name.endswith(".png"):
                image_paths.append(os.path.join(img_folder, img_name))
                label_indices.append(label_map[category])

# ---------- Transform, Dataset, Dataloader ----------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

dataset = RGBDDataset(image_paths, label_indices, transform=transform)

train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# ---------- Model ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(label_map))
model.to(device)

# ---------- Training ----------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 5

print("Training CNN...")
model.train()
for epoch in range(epochs):
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")

# ---------- Evaluation ----------
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# ---------- Report ----------
print("CNN Classification Report:")
print(classification_report(all_labels, all_preds, target_names=categories))
print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

# Save the model
# Ensure the results folder exists
os.makedirs("results", exist_ok=True)

# Save CNN classification report and confusion matrix to a file
with open("results/cnn_results.txt", "w") as f:
    report = classification_report(all_labels, all_preds, target_names=categories)
    matrix = confusion_matrix(all_labels, all_preds)

    f.write("CNN Classification Report:\n")
    f.write(report + "\n\n")
    
    f.write("Confusion Matrix:\n")
    for row in matrix:
        f.write(" ".join(map(str, row)) + "\n")

print("✅ CNN results saved to results/cnn_results.txt")

