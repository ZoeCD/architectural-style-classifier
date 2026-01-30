import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from collections import Counter
from pathlib import Path
from tqdm import tqdm
import os
import json

# --------- Configuration ---------

DATA_PATH = Path("../data/merged")
MODEL_SAVE_PATH = Path("./models")
MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 1e-3
FINE_TUNE_LR = 1e-4
FREEZE_EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# --------- Data Trasformation ---------

train_trasnsforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --------- Load Dataset ---------

full_dataset = datasets.ImageFolder(DATA_PATH, transform=train_trasnsforms)
class_names = full_dataset.classes

print(f"Found {len(full_dataset)} images belonging to {len(class_names)} classes.")

# Save class labels
with open(MODEL_SAVE_PATH / "class_labels.json", "w") as f:
    json.dump(class_names, f)

# --------- Split Dataset ---------
train_size = int(0.7 * len(full_dataset))
val_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset, 
    [train_size, val_size, test_size], 
    generator = torch.Generator().manual_seed(42)
)

val_dataset.dataset.transform = val_transforms
test_dataset.dataset.transform = val_transforms

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

# --------- Calculate class weights to handle imbalance data ---------

class_counts = Counter([full_dataset.targets[i] for i in train_dataset.indices])
total_samples = sum(class_counts.values())
class_weights = torch.tensor([
    total_samples / class_counts[i] for i in range(len(class_names)) # The more frequent the class, the smaller the weight
], dtype=torch.float).to(DEVICE)

# Normalize weights
class_weights = class_weights / class_weights.mean()

print(f"Class weights range: {class_weights.min().item()} - {class_weights.max().item()}")

# --------- Model Setup ---------

model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

# Freeze all layers initially
for param in model.parameters():
    param.requires_grad = False

# Recplace the classifier head
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, len(class_names))

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights)


# --------- Training Functions ---------
def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating", leave=False):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# --------- Training Loop ---------
best_val_acc = 0.0

# Train only the classifier head first
print("Training classifier head...")
optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)

for epoch in range(FREEZE_EPOCHS):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc = validate(model, val_loader, criterion)

    print(f"Epoch [{epoch+1}/{FREEZE_EPOCHS}] "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_SAVE_PATH / "best_model.pth")
        print("Best model saved. (val acc: {:.4f})".format(best_val_acc))

# Unfreeze all layers for fine-tuning
print("Fine-tuning entire model...")
for param in model.parameters():
    param.requires_grad = True

optimizer = optim.Adam(model.parameters(), lr=FINE_TUNE_LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

remaining_epochs = NUM_EPOCHS - FREEZE_EPOCHS

for epoch in range(remaining_epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc = validate(model, val_loader, criterion)

    scheduler.step(val_acc)
    current_lr = optimizer.param_groups[0]['lr']

    print(f"Epoch [{epoch+1}/{remaining_epochs}] "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | LR: {current_lr:.6f}")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_SAVE_PATH / "best_model.pth")
        print("Best model saved. (val acc: {:.4f})".format(best_val_acc))
    
# Save final model
torch.save(model.state_dict(), MODEL_SAVE_PATH / "final_model.pth")
print("Training complete. Final model saved.")
print(f"Models are saved in: {MODEL_SAVE_PATH}/")

# --------- Test Evaluation ---------
print("Evaluating on test set...")

# Load the best model
model.load_state_dict(torch.load(MODEL_SAVE_PATH / "best_model.pth"))

test_loss, test_acc = validate(model, test_loader, criterion)

print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

# Save result
with open(MODEL_SAVE_PATH / "test_results.json", "w") as f:
    json.dump({
        "test_loss": test_loss,
        "test_accuracy": test_acc, 
        "num_test_samples": len(test_dataset)
    }, f, indent=2)

print("Test results saved.")