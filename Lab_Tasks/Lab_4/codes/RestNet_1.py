import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from pathlib import Path
from PIL import Image

# ==========================
# CONFIGURATION
# ==========================
train_dir = 'Step1Training'
test_dir = 'Step1Testing'
img_size = 224  # ResNet50 default
batch_size = 16
epochs = 10
model_path = 'resnet50_model.pth'
output_dir = Path('resnet50_test')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================
# TRANSFORMS
# ==========================
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Required for pretrained ResNet
])

# ==========================
# DATA LOADERS
# ==========================
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.ImageFolder(test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

class_names = train_dataset.classes
print(f"Classes: {class_names}")

# ==========================
# RESNET50 MODEL
# ==========================
resnet = models.resnet50(pretrained=True)

# Freeze feature extractor layers
for param in resnet.parameters():
    param.requires_grad = False

# Replace classifier head
num_ftrs = resnet.fc.in_features
resnet.fc = nn.Sequential(
    nn.Linear(num_ftrs, 128),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(128, 2)
)
resnet = resnet.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet.fc.parameters(), lr=0.001)

# ==========================
# TRAINING
# ==========================
print("Training ResNet50...")
for epoch in range(epochs):
    resnet.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = resnet(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {running_loss/len(train_loader):.4f}")

# ==========================
# SAVE MODEL
# ==========================
torch.save(resnet.state_dict(), model_path)
print(f"Model saved to {model_path}")

# ==========================
# EVALUATION
# ==========================
print("Evaluating on test set...")
resnet.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = resnet(inputs)
        _, preds = torch.max(outputs, 1)
        y_true.append(labels.item())
        y_pred.append(preds.item())

acc = accuracy_score(y_true, y_pred)
print(f"Test Accuracy: {acc * 100:.2f}% ({sum([a == b for a, b in zip(y_true, y_pred)])}/20)")

# ==========================
# SAVE PREDICTED IMAGES
# ==========================
print("Saving predicted images to resnet50_test/...")

# Get original file paths
original_paths = test_dataset.samples

# Create output folders
for class_name in class_names:
    (output_dir / class_name).mkdir(parents=True, exist_ok=True)

# Save images in predicted folders
for idx, (inputs, _) in enumerate(test_loader):
    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = resnet(inputs)
        _, predicted = torch.max(outputs, 1)

    img_path, _ = original_paths[idx]
    img = Image.open(img_path).convert("RGB")
    pred_class = class_names[predicted.item()]
    filename = os.path.basename(img_path)
    save_path = output_dir / pred_class / filename
    img.save(save_path)

print("Done! Predicted images saved in resnet50_test/")
