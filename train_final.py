"""
Script expérimental de classification avec MobileNetV2.
Ce script a été utilisé lors des phases de test pour évaluer 
l'architecture MobileNetV2 (très légère et rapide) avant de 
basculer sur ResNet50 pour maximiser la précision.
"""
import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms, models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from dataset_7classes import GeoRock7ClassesDataset

def get_dataloaders(data_dir, batch_size=16):
    full_dataset = GeoRock7ClassesDataset(root_dir=data_dir)
    
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        full_dataset.file_paths, full_dataset.labels, test_size=0.2, stratify=full_dataset.labels, random_state=42
    )
    
    # Enhanced Data Augmentation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dataset = GeoRock7ClassesDataset(root_dir=data_dir, transform=train_transform, file_paths=train_paths, labels=train_labels, class_to_idx=full_dataset.class_to_idx)
    val_dataset = GeoRock7ClassesDataset(root_dir=data_dir, transform=val_transform, file_paths=val_paths, labels=val_labels, class_to_idx=full_dataset.class_to_idx)
    
    # Weighted Random Sampler
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    sample_weights = np.array([class_weights[label] for label in train_labels])
    sampler = WeightedRandomSampler(weights=torch.from_numpy(sample_weights).double(), num_samples=len(sample_weights), replacement=True)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, full_dataset.classes

def create_model(num_classes):
    print("Loading MobileNetV2...")
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    
    # Freeze the early layers, unfreeze the later layers for fine-tuning
    # MobileNetV2 features contain 19 blocks (0 to 18)
    for param in model.features.parameters():
        param.requires_grad = False
        
    # Unfreeze blocks 14 to 18
    for param in model.features[14:].parameters():
        param.requires_grad = True

    # Replace classifier
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model

def plot_results(history, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'bo-', label='Training loss')
    plt.plot(epochs, history['val_loss'], 'ro-', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'bo-', label='Training acc')
    plt.plot(epochs, history['val_acc'], 'ro-', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'))
    plt.close()

def main():
    data_dir = 'data/Dataset'
    output_dir = 'training_results'
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_loader, val_loader, classes_dict = get_dataloaders(data_dir, batch_size=16)
    dataloaders = {'train': train_loader, 'val': val_loader}
    
    # classes_dict is a dict mapping name to idx in the dataset logic we wrote earlier, wait!
    # Let's rebuild the ordered class names list
    class_names = ['Basalt', 'Coal', 'Granite', 'Limestone', 'Marble', 'Quartzite', 'Sandstone']
    num_classes = len(class_names)
    
    model = create_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Using AdamW which handles weight decay better
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
    
    # Learning Rate Scheduler (Reduces LR when validation loss stops improving)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Training Parameters
    num_epochs = 30
    early_stopping_patience = 6
    epochs_no_improve = 0
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    print("Starting Training...")
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        for phase in ['train', 'val']:
            if phase == 'train': model.train()
            else: model.eval()
                
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())
            
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            if phase == 'val':
                scheduler.step(epoch_loss)
                
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                    print("--> New best model saved!")
                    torch.save(model.state_dict(), 'best_georock_model_7classes.pth')
                    with open('best_model_arch.txt', 'w') as f:
                        f.write('MobileNetV2')
                else:
                    epochs_no_improve += 1
                    
        if epochs_no_improve >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs.")
            break
            
    print(f"\nTraining Complete. Best Validation Accuracy: {best_acc:.4f}")
    plot_results(history, output_dir)
    
    # ─── Evaluate Final Model ───
    model.load_state_dict(best_model_wts)
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

    # Classification Report
    report = classification_report(all_labels, all_preds, target_names=class_names)
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
        
    print("\nFinal classification report generated.")

if __name__ == '__main__':
    main()
