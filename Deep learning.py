import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import yaml, logging, os
from tqdm import tqdm

# ----------------------------
# Load Configuration
# ----------------------------
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# Logging setup
os.makedirs(cfg["logging"]["log_dir"], exist_ok=True)
logging.basicConfig(
    filename=os.path.join(cfg["logging"]["log_dir"], "training.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# Device setup
device = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# ----------------------------
# Data Transformations
# ----------------------------
transform = {
    "train": transforms.Compose([
        transforms.Resize((cfg["dataset"]["image_size"], cfg["dataset"]["image_size"])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    "val": transforms.Compose([
        transforms.Resize((cfg["dataset"]["image_size"], cfg["dataset"]["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

train_data = datasets.ImageFolder(cfg["dataset"]["train_dir"], transform["train"])
val_data = datasets.ImageFolder(cfg["dataset"]["val_dir"], transform["val"])

train_loader = DataLoader(train_data, batch_size=cfg["training"]["batch_size"], shuffle=True)
val_loader = DataLoader(val_data, batch_size=cfg["training"]["batch_size
