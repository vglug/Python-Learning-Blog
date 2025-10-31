import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# --------------------------
# 1. Configuration
# --------------------------
DATA_DIR = "data"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create dummy folders if missing
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)
print("Data folders checked/created.")

# --------------------------
# 2. Data Transforms
# --------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

try:
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform)
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False)
    class_names = train_dataset.classes
    print(f"Detected classes: {class_names}")
except Exception as e:
    print("Error loading datasets:", e)
    print("Make sure you have structure: data/train/class0/... , data/val/class0/...")
    raise

# --------------------------
# 3. Define CNN Model
# --------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 56 * 56, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

num_classes = len(train_dataset.classes)
model = SimpleCNN(num_classes).to(DEVICE)

# --------------------------
# 4. Training Setup
# --------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --------------------------
# 5. Training Loop
# --------------------------
best_val_acc = 0.0
for epoch in range(1, 6):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

    train_acc = 100 * correct / total

    # Validation
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (preds == labels).sum().item()
    val_acc = 100 * val_correct / val_total if val_total > 0 else 0

    print(f"Epoch [{epoch}/5] | Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")
        print("Best model saved.")

print(f"Best Validation Accuracy: {best_val_acc:.2f}%")

# --------------------------
# 6. Grad-CAM Implementation
# --------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
            if output.requires_grad:
                output.register_hook(self.save_gradient)

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def save_gradient(self, grad):
        self.gradients = grad.detach()

    def generate_heatmap(self, input_tensor, class_idx=None):
        self.model.eval()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        self.model.zero_grad()
        target = output[0, class_idx]
        target.backward()

        if self.gradients is None:
            raise RuntimeError("Gradients not captured. Check hooks or target_layer.")

        grads = self.gradients
        acts = self.activations
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        cam /= cam.max()
        return cam

# --------------------------
# 7. Visualization
# --------------------------
def visualize_gradcam(image_path):
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)

    gradcam = GradCAM(model, model.features[-3])
    heatmap = gradcam.generate_heatmap(tensor)

    plt.imshow(img)
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    plt.axis("off")
    plt.title("Grad-CAM Visualization")
    plt.show()

# --------------------------
# 8. Test Example
# --------------------------
test_dir = os.path.join(DATA_DIR, "val")
test_images = []
for root, _, files in os.walk(test_dir):
    for f in files:
        if f.lower().endswith(('.jpg', '.png')):
            test_images.append(os.path.join(root, f))

if test_images:
    test_image = random.choice(test_images)
    print("Visualizing Grad-CAM for:", test_image)
    visualize_gradcam(test_image)
else:
    print("No test images found in 'data/val'. Please add images.")
