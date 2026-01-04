import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import accuracy_score
from pathlib import Path
from PIL import Image

# =======================
# CONFIG
# =======================
train_dir = 'Step1Training'
test_dir = 'Step1Testing'
img_size = 64
batch_size = 16
epochs = 10
model_path = 'cnn.pth'
output_dir = Path('cnn_test')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =======================
# TRANSFORMS
# =======================
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
])

# =======================
# DATASETS & LOADERS
# =======================
train_dataset = ImageFolder(root=train_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = ImageFolder(root=test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)  # batch size 1 for prediction

# =======================
# CLASS LABELS
# =======================
class_names = train_dataset.classes
print(f"Classes: {class_names}")  # Should be ['no rust', 'rust']

# =======================
# CNN MODEL
# =======================
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(32 * (img_size // 4) * (img_size // 4), 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.network(x)

model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# =======================
# TRAINING
# =======================
print("Training...")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {running_loss/len(train_loader):.4f}")

# =======================
# SAVE MODEL
# =======================
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# =======================
# EVALUATE
# =======================
print("Evaluating on test set...")
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        y_true.append(labels.item())
        y_pred.append(preds.item())

acc = accuracy_score(y_true, y_pred)
print(f"Test Accuracy: {acc * 100:.2f}% ({sum([a == b for a, b in zip(y_true, y_pred)])}/20)")

# =======================
# SAVE PREDICTED IMAGES
# =======================
print("Saving predicted images to cnn_test/...")

# Re-load test data with paths
test_dataset = ImageFolder(root=test_dir, transform=transform)
original_paths = test_dataset.samples  # List of (path, label)

# Make sure output folders exist
for class_name in class_names:
    (output_dir / class_name).mkdir(parents=True, exist_ok=True)

# Save each image in predicted folder
for idx, (inputs, _) in enumerate(test_loader):
    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

    # Original image path
    img_path, _ = original_paths[idx]
    img = Image.open(img_path).convert("RGB")
    pred_class = class_names[predicted.item()]
    filename = os.path.basename(img_path)
    save_path = output_dir / pred_class / filename
    img.save(save_path)

print("Done! Predicted images saved in cnn_test/.")
