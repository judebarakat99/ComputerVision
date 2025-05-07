import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

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

# ---------- Deep CNN Model ----------
class DeepCNN(nn.Module):
    def __init__(self, num_classes):
        super(DeepCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.25)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(64 * 32 * 32, 512)  # Based on 128x128 input, after 2 pools
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout1(x)

        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.dropout1(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

# ---------- Training Setup ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepCNN(num_classes=len(label_map)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters(), lr=0.001)
epochs = 40

# ---------- Training Loop with History ----------
history = {"loss": [], "accuracy": []}
print("Training Deep CNN...")
model.train()
for epoch in range(epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / len(train_loader)
    accuracy = correct / total
    history["loss"].append(avg_loss)
    history["accuracy"].append(accuracy)

    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Accuracy: {accuracy*100:.2f}%")

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
print("\nDeep CNN Classification Report:")
print(classification_report(all_labels, all_preds, target_names=categories))
print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

# ---------- Plot Training History ----------
plt.figure(figsize=(10, 4))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history['accuracy'], label='Train Accuracy')
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history['loss'], label='Train Loss', color='orange')
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
os.makedirs("graphs", exist_ok=True)
plt.savefig("graphs/deep_cnn_training_plots.png")
plt.show()

# ---------- Save Classification Report ----------
with open("results/deep_cnn_results.txt", "w") as f:
    report = classification_report(all_labels, all_preds, target_names=categories)
    matrix = confusion_matrix(all_labels, all_preds)

    f.write("Deep CNN Classification Report:\n")
    f.write(report + "\n\n")

    f.write("Confusion Matrix:\n")
    for row in matrix:
        f.write(" ".join(map(str, row)) + "\n")

print("✅ Deep CNN results and training plots saved to 'results/'")
