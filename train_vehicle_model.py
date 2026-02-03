"""
Vehicle Detection Model Training Script
Trains ResNet18 on local vehicle dataset to 90%+ accuracy
Exports trained model to ONNX format for Java inference
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchmetrics import Accuracy, ConfusionMatrix
import os
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from random import shuffle, seed
import json

# Configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MANUAL_SEED = 42
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 30
IMAGE_SIZE = 224
DATASET_ROOT = "NonVehicleDetectionSystem/dataset/raw"
MODEL_OUTPUT_DIR = "NonVehicleDetectionSystem/model"
ACCURACY_TARGET = 0.90

print(f"Device: {DEVICE}")
print(f"PyTorch Version: {torch.__version__}")

# ============================================================================
# DATASET CLASS
# ============================================================================
class VehicleDataset(Dataset):
    def __init__(self, root, split='train', transform=transforms.ToTensor()):
        self.root = Path(root)
        self.transform = transform
        self.split = split
        
        # Get all vehicle type folders
        self.classes = sorted([d.name for d in self.root.glob('*') if d.is_dir() and d.name != '__pycache__'])
        print(f"Classes found: {self.classes}")
        
        self.data = []
        for class_idx, class_name in enumerate(self.classes):
            class_dir = self.root / class_name
            if not class_dir.exists():
                continue
            
            # Support nested folder structure (e.g., Vehicles/Cars/car_001.jpg)
            image_files = list(class_dir.glob("**/*.jpg")) + list(class_dir.glob("**/*.png"))
            print(f"Class '{class_name}': {len(image_files)} images")
            
            for img_path in image_files:
                self.data.append((img_path, class_idx))
        
        print(f"Total images found: {len(self.data)}")
        
        # Split data into train/val/test
        train_data = []
        val_data = []
        test_data = []
        
        for class_idx in range(len(self.classes)):
            class_images = [(p, l) for p, l in self.data if l == class_idx]
            shuffle(class_images)
            
            n_total = len(class_images)
            n_train = int(0.7 * n_total)
            n_val = int(0.15 * n_total)
            
            train_data.extend(class_images[:n_train])
            val_data.extend(class_images[n_train:n_train + n_val])
            test_data.extend(class_images[n_train + n_val:])
        
        data_split = {'train': train_data, 'validation': val_data, 'test': test_data}
        self.data = data_split[split]
        
        print(f"{split.upper()} split size: {len(self.data)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# ============================================================================
# DATA LOADING
# ============================================================================
print("\n" + "="*70)
print("LOADING DATASET")
print("="*70)

seed(MANUAL_SEED)
torch.manual_seed(MANUAL_SEED)

# Transform pipeline
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet normalization
        std=[0.229, 0.224, 0.225]
    ),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.33))
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Create datasets
train_dataset = VehicleDataset(DATASET_ROOT, split='train', transform=train_transform)
val_dataset = VehicleDataset(DATASET_ROOT, split='validation', transform=val_transform)
test_dataset = VehicleDataset(DATASET_ROOT, split='test', transform=val_transform)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

num_classes = len(train_dataset.classes)
print(f"\nNumber of classes: {num_classes}")
print(f"Classes: {train_dataset.classes}")

# ============================================================================
# MODEL SETUP
# ============================================================================
print("\n" + "="*70)
print("SETTING UP MODEL")
print("="*70)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)
model = model.to(DEVICE)

print(f"Model: ResNet18 with {num_classes} output classes")

# Loss, optimizer, scheduler
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
accuracy_metric = Accuracy(task="multiclass", num_classes=num_classes).to(DEVICE)

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================
def train_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    
    pbar = tqdm(loader, desc="Training")
    for X, y in pbar:
        X, y = X.to(device), y.to(device)
        
        logits = model(X)
        loss = loss_fn(logits, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        preds = torch.argmax(logits, dim=1)
        acc = accuracy_metric(preds, y)
        
        total_loss += loss.item()
        total_acc += acc.item()
        
        pbar.set_postfix({'loss': loss.item(), 'acc': acc.item()})
    
    avg_loss = total_loss / len(loader)
    avg_acc = total_acc / len(loader)
    return avg_loss, avg_acc

def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="Validating")
        for X, y in pbar:
            X, y = X.to(device), y.to(device)
            
            logits = model(X)
            loss = loss_fn(logits, y)
            preds = torch.argmax(logits, dim=1)
            acc = accuracy_metric(preds, y)
            
            total_loss += loss.item()
            total_acc += acc.item()
            
            pbar.set_postfix({'loss': loss.item(), 'acc': acc.item()})
    
    avg_loss = total_loss / len(loader)
    avg_acc = total_acc / len(loader)
    return avg_loss, avg_acc

# ============================================================================
# TRAINING LOOP
# ============================================================================
print("\n" + "="*70)
print("TRAINING MODEL")
print("="*70)

best_val_acc = 0.0
best_model_state = None
train_losses, val_losses = [], []
train_accs, val_accs = [], []

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
    
    train_loss, train_acc = train_epoch(model, train_loader, loss_fn, optimizer, DEVICE)
    val_loss, val_acc = validate(model, val_loader, loss_fn, DEVICE)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
    print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = model.state_dict().copy()
        print(f"✓ New best validation accuracy: {val_acc*100:.2f}%")
    
    scheduler.step()
    
    # Early stopping if target reached
    if val_acc >= ACCURACY_TARGET:
        print(f"\n🎯 TARGET ACCURACY REACHED: {val_acc*100:.2f}%")
        break

# ============================================================================
# LOAD BEST MODEL AND TEST
# ============================================================================
print("\n" + "="*70)
print("TESTING BEST MODEL")
print("="*70)

if best_model_state is not None:
    model.load_state_dict(best_model_state)

test_loss, test_acc = validate(model, test_loader, loss_fn, DEVICE)
print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc*100:.2f}%")

# ============================================================================
# SAVE MODEL AND METADATA
# ============================================================================
print("\n" + "="*70)
print("SAVING MODEL")
print("="*70)

output_dir = Path(MODEL_OUTPUT_DIR)
output_dir.mkdir(parents=True, exist_ok=True)

# Save PyTorch model
pth_path = output_dir / "vehicle_detector.pth"
torch.save({
    'model_state_dict': model.state_dict(),
    'classes': train_dataset.classes,
    'accuracy': test_acc,
}, pth_path)
print(f"✓ PyTorch model saved: {pth_path}")

# Save class labels
labels_path = output_dir / "labels.txt"
with open(labels_path, 'w') as f:
    for cls in train_dataset.classes:
        f.write(cls + '\n')
    f.write('non_vehicle\n')
print(f"✓ Labels saved: {labels_path}")

# Save model config
config_path = output_dir / "model_config.json"
config = {
    "model_type": "ResNet18",
    "num_classes": num_classes,
    "input_size": IMAGE_SIZE,
    "classes": train_dataset.classes + ["non_vehicle"],
    "test_accuracy": float(test_acc),
    "validation_accuracy": float(best_val_acc)
}
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)
print(f"✓ Model config saved: {config_path}")

# ============================================================================
# EXPORT TO ONNX
# ============================================================================
print("\n" + "="*70)
print("EXPORTING TO ONNX")
print("="*70)

model.eval()
dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)

onnx_path = output_dir / "vehicle_detector.onnx"
torch.onnx.export(
    model,
    dummy_input,
    str(onnx_path),
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
    verbose=False
)
print(f"✓ ONNX model exported: {onnx_path}")

print("\n" + "="*70)
print("✅ TRAINING COMPLETE!")
print("="*70)
print(f"Best Validation Accuracy: {best_val_acc*100:.2f}%")
print(f"Test Accuracy: {test_acc*100:.2f}%")
print(f"Classes: {', '.join(train_dataset.classes)}")
print(f"ONNX Model: {onnx_path}")
print("="*70)
