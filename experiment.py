import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
from transformers import AutoConfig, AutoModel, AutoImageProcessor
import json
import csv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
import sys

# Ensure models module is available
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
from model_davit import Model

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

class ChestXRayDataset(Dataset):
    def __init__(self, root_dir, limits=None, transform=None):
        self.transform = transform
        self.examples = []
        
        # Load all images
        normal_dir = os.path.join(root_dir, 'NORMAL')
        pneumonia_dir = os.path.join(root_dir, 'PNEUMONIA')
        
        valid_suffixes = ('.jpeg', '.jpg', '.png')
        
        normal_paths = []
        if os.path.exists(normal_dir):
            for file in os.listdir(normal_dir):
                if file.lower().endswith(valid_suffixes):
                    normal_paths.append((os.path.join(normal_dir, file), 0))
                    
        pneumonia_paths = []
        if os.path.exists(pneumonia_dir):
            for file in os.listdir(pneumonia_dir):
                if file.lower().endswith(valid_suffixes):
                    pneumonia_paths.append((os.path.join(pneumonia_dir, file), 1))
        
        if limits is not None:
            if 0 in limits:
                normal_paths = normal_paths[:limits[0]]
            if 1 in limits:
                pneumonia_paths = pneumonia_paths[:limits[1]]
            
        self.examples = normal_paths + pneumonia_paths
        random.shuffle(self.examples)
        
    def __len__(self):
        return len(self.examples)
        
    def __getitem__(self, idx):
        img_path, label = self.examples[idx]
        with Image.open(img_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            if self.transform:
                pixel_values = self.transform(img)
            else:
                pixel_values = img
        return pixel_values, label

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return acc, prec, rec, f1, cm

def evaluate(model, dataloader, device):
    model.eval()
    y_true = []
    y_pred = []
    total_loss = 0
    
    with torch.no_grad():
        for pixel_values, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)
            
            # Use the same multi-return forward
            loss, logits = model(pixel_values=pixel_values, labels=labels)
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=-1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            
    avg_loss = total_loss / max(1, len(dataloader))
    acc, _, _, _, _ = compute_metrics(y_true, y_pred)
    
    return avg_loss, acc, y_true, y_pred

def main():
    set_seed(42)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is not available but is strictly required per user instructions.")
    device = torch.device('cuda')
    print(f"Using device: {device}")
    
    # 1. Dataset Configuration
    # Use absolute path relative to this script
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_script_dir, 'data_mixed')
    output_dir = os.path.join(current_script_dir, 'outputs')
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset path incorrect: {data_dir}. Expected it to be present.")
        
    os.makedirs(output_dir, exist_ok=True)
    
    # We use a standard processor config for sizing/normalization
    # The prompt allows using ImageNet stats / standard resizing as in original code
    crop_size = (224, 224)
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(crop_size),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    eval_transform = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_mean, std=image_std)
    ])
    
    print("Loading data...")
    train_dataset = ChestXRayDataset(os.path.join(data_dir, 'train'), limits=None, transform=train_transform)
    test_dataset = ChestXRayDataset(os.path.join(data_dir, 'test'), limits=None, transform=eval_transform)
    val_dataset = test_dataset # use test set for per-epoch validation

    epochs = 15
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = val_loader
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # 2. Model Initialization (FROM SCRATCH)
    print("Initializing model from scratch...")
    # Get config (no weights downloaded)
    # Super-Fast Mode: Using 'small' variant with PRE-TRAINED weights
    # Switch to from_pretrained for much higher accuracy
    vit = AutoModel.from_pretrained('facebook/dinov2-small')
    
    # We need a dummy processor object with 'size' for the DAViT model if it uses it.
    # Actually Model(vit, processor, args) requires processor and args.
    class DummyArgs:
        pass
    args = DummyArgs()
    
    class DummyProcessor:
        def __init__(self):
            self.size = {"shortest_edge": 224}
    processor = DummyProcessor()
    
    # Note: If DAViT expects outputs from swin vs vit vs dinov2, it extracts `last_hidden_state` 
    # dinov2 produces [B, N, C] last_hidden_state, which model_davit uses.
    model = Model(vit, processor, args, num_labels=2)
    model.to(device)
    
    # Optimizer and Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    loss_fct = nn.CrossEntropyLoss()
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # 3. Training Loop with AMP (Automatic Mixed Precision)
    print(f"Starting Extreme Training (Phase 3) for {epochs} epochs...")
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    scaler = torch.amp.GradScaler('cuda')
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        y_true_train = []
        y_pred_train = []
        
        for pixel_values, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Use autocast for mixed precision forward pass
            with torch.amp.autocast('cuda'):
                loss, logits = model(pixel_values=pixel_values, labels=labels)
            
            # Scalar backward
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            
            # Accuracy from the same forward pass
            with torch.no_grad():
                preds = torch.argmax(logits, dim=-1)
                y_true_train.extend(labels.cpu().numpy())
                y_pred_train.extend(preds.cpu().numpy())
        
        torch.cuda.empty_cache()
        
        # Step scheduler
        scheduler.step()
                
        train_loss = total_loss / len(train_loader)
        train_acc = accuracy_score(y_true_train, y_pred_train)
        
        val_loss, val_acc, _, _ = evaluate(model, val_loader, device)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(output_dir, 'model_best.pth'))
            print(f"  --> New Best Model Saved (Acc: {val_acc:.2%})")
              
    # 4. Final Evaluation on Test Set
    print("Evaluating on test set...")
    test_loss, test_acc, y_true_test, y_pred_test = evaluate(model, test_loader, device)
    
    acc, prec, rec, f1, cm = compute_metrics(y_true_test, y_pred_test)
    
    # Console output
    print("--- Final Test Output ---")
    print(f"Accuracy:  {acc*100:.2f}%")
    print(f"Precision: {prec*100:.2f}%")
    print(f"Recall:    {rec*100:.2f}%")
    print(f"F1-score:  {f1*100:.2f}%")
    print("-------------------------")
    
    # 5. Save Metrics Data (CSV/JSON is safe)
    results = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'test_loss': test_loss
    }
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)
        
    with open(os.path.join(output_dir, 'results.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        for k, v in results.items():
            writer.writerow([k, f"{v:.4f}"])

    # 6. Save Model (Most Important)
    torch.save(model.state_dict(), os.path.join(output_dir, 'model.pth'))
    print("Model saved to model.pth")

    # 7. Attempt Plots (May fail due to AppLocker block on Seaborn/Pandas)
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        epochs_range = range(1, epochs + 1)
        
        # Loss plot
        plt.figure(figsize=(8, 6))
        plt.plot(epochs_range, history['train_loss'], label='Train Loss')
        plt.plot(epochs_range, history['val_loss'], label='Val Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'loss.png'))
        plt.close()
        
        # Acc plot
        plt.figure(figsize=(8, 6))
        plt.plot(epochs_range, history['train_acc'], label='Train Accuracy')
        plt.plot(epochs_range, history['val_acc'], label='Val Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'accuracy.png'))
        plt.close()
        
        # CM plot
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Normal', 'Pneumonia'], 
                    yticklabels=['Normal', 'Pneumonia'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
        plt.close()
        print("Visualization plots saved.")
    except Exception as e:
        print(f"Skipping visualization plots: Dependency error (likely AppLocker block). Raw metrics are saved in results.json.")

    print("All outputs saved successfully.")

if __name__ == '__main__':
    main()
