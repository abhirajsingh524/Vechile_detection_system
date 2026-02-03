import argparse
import os
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torchmetrics import Accuracy
from tqdm.auto import tqdm


def make_transforms(img_size=224):
    train_t = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.1, 0.1, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4641, 0.4725, 0.4705], std=[0.3169, 0.3097, 0.3201])
    ])

    val_t = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4641, 0.4725, 0.4705], std=[0.3169, 0.3097, 0.3201])
    ])
    return train_t, val_t


def build_loaders(dataset_dir, batch_size, img_size, seed=42, num_workers=0):
    # Support two layouts:
    # 1) dataset_dir/<class>/*.jpg  (ImageFolder standard)
    # 2) dataset_dir/vehicle/<subclass>/**/*.jpg and optional dataset_dir/non_vehicle images
    dataset_dir = Path(dataset_dir)
    candidate_vehicle = dataset_dir / 'vehicle'
    if candidate_vehicle.exists() and any(candidate_vehicle.iterdir()):
        # use subclasses under vehicle/ as classes
        print('Detected nested vehicle folder layout; using vehicle/* as classes')
        full = ImageFolder(root=str(candidate_vehicle))
    else:
        full = ImageFolder(root=str(dataset_dir))

    classes = full.classes
    if len(classes) < 2:
        raise RuntimeError("Dataset must contain at least 2 classes")

    # We create deterministic splits (70/15/15)
    total = len(full)
    n_train = int(0.7 * total)
    n_val = int(0.15 * total)
    n_test = total - n_train - n_val

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(full, [n_train, n_val, n_test], generator=generator)

    # Apply transforms (ImageFolder carries transform via dataset.transform)
    train_t, val_t = make_transforms(img_size)
    train_set.dataset.transform = train_t
    val_set.dataset.transform = val_t
    test_set.dataset.transform = val_t

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, classes


def create_model(num_classes, backbone='resnet18', pretrained=True):
    if backbone == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        raise NotImplementedError('backbone not supported')
    return model


def train_one_epoch(model, loader, criterion, optimiser, device):
    model.train()
    running_loss = 0.0
    acc_metric = Accuracy(task='multiclass', num_classes=model.fc.out_features).to(device)

    for X, y in tqdm(loader, desc='train', leave=False):
        X, y = X.to(device), y.to(device)
        optimiser.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimiser.step()

        running_loss += loss.item() * X.size(0)
        preds = torch.argmax(logits, dim=1)
        acc_metric.update(preds, y)

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = acc_metric.compute().item()
    return epoch_loss, epoch_acc


@torch.inference_mode()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    acc_metric = Accuracy(task='multiclass', num_classes=model.fc.out_features).to(device)

    for X, y in tqdm(loader, desc='val', leave=False):
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = criterion(logits, y)
        running_loss += loss.item() * X.size(0)
        preds = torch.argmax(logits, dim=1)
        acc_metric.update(preds, y)

    val_loss = running_loss / len(loader.dataset)
    val_acc = acc_metric.compute().item()
    return val_loss, val_acc


def save_checkpoint(model, classes, out_dir, epoch, val_acc):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    model_path = out_dir / f'best_model_{timestamp}.pth'
    torch.save({'model_state_dict': model.state_dict(), 'classes': classes, 'val_acc': val_acc, 'epoch': epoch}, model_path)
    # also write metadata
    meta = {
        'trained_on': timestamp,
        'val_accuracy': float(val_acc),
        'classes': classes
    }
    import json
    with open(out_dir / 'model_metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)
    return model_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='NonVehicleDetectionSystem/dataset/raw', help='path to dataset root (subfolders per class)')
    parser.add_argument('--epochs', type=int, default=2, help='number of epochs (smoke default=2)')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--out', default='models')
    parser.add_argument('--backbone', default='resnet18')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--num-workers', type=int, default=0, help='dataloader num_workers (0 for Windows)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    train_loader, val_loader, test_loader, classes = build_loaders(args.dataset, args.batch_size, args.img_size, num_workers=args.num_workers)
    print('Classes:', classes)

    model = create_model(len(classes), backbone=args.backbone)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val = 0.0
    patience = 5
    wait = 0

    for epoch in range(1, args.epochs + 1):
        print(f'Epoch {epoch}/{args.epochs}')
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimiser, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        t1 = time.time()
        print(f'Time: {t1 - t0:.1f}s  Train loss: {train_loss:.4f} acc: {train_acc*100:.2f}%  Val loss: {val_loss:.4f} acc: {val_acc*100:.2f}%')

        if val_acc > best_val:
            best_val = val_acc
            path = save_checkpoint(model, classes, args.out, epoch, best_val)
            print('Saved best model:', path)
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print('Early stopping')
                break

        # quick exit for smoke runs
        if args.smoke:
            print('Smoke run flag detected, finishing after one epoch loop.')
            break

    # final test evaluation
    print('Evaluating on test set...')
    # load best checkpoint if exists
    # for simplicity evaluate current model
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f'Test loss: {test_loss:.4f}  Test acc: {test_acc*100:.2f}%')

    # export to ONNX using helper script if accuracy good


if __name__ == '__main__':
    main()
