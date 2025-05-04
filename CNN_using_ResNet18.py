import torch
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from PIL import Image

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

# Load image paths and labels for torch
image_paths = []
label_indices = []
label_map = {cat: i for i, cat in enumerate(categories[:5])}
for i, category in enumerate(categories[:5]):
    cat_path = os.path.join(DATASET_PATH, category)
    for video_folder in os.listdir(cat_path)[:1]:
        img_folder = os.path.join(cat_path, video_folder)
        for img_name in os.listdir(img_folder)[:30]:
            if img_name.endswith(".png"):
                image_paths.append(os.path.join(img_folder, img_name))
                label_indices.append(label_map[category])

# Define transform and dataloader
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

dataset = RGBDDataset(image_paths, label_indices, transform=transform)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Load pretrained model
model = models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, len(label_map))  # adjust to 5 classes

# Fine-tuning (not fully shown here, but you can train using typical PyTorch training loop)
