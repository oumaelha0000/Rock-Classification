"""
Script principal d'entraînement utilisant l'architecture ResNet50.
Ce script inclut :
- Le chargement des données avec une Data Augmentation extrême.
- L'utilisation d'un WeightedRandomSampler pour gérer le déséquilibre des classes.
- Le Fine-Tuning du ResNet50 (remplacement de la dernière couche).
- La génération des graphiques de performance et de la matrice de confusion.
"""
import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataset_7classes import GeoRock7ClassesDataset

def get_dataloaders(data_dir, batch_size=16):
    """
    Prépare les DataLoaders pour l'entraînement et la validation.
    Applique une augmentation de données sévère pour forcer le modèle 
    à apprendre les textures sous divers angles et conditions lumineuses.
    """
    # Data Augmentation (Extrême)
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    full_dataset = GeoRock7ClassesDataset(root_dir=data_dir)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds_raw, val_ds_raw = random_split(
        full_dataset, 
        [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )

    train_dataset = GeoRock7ClassesDataset(
        root_dir=data_dir, 
        transform=transform_train,
        file_paths=[full_dataset.file_paths[i] for i in train_ds_raw.indices],
        labels=[full_dataset.labels[i] for i in train_ds_raw.indices],
        class_to_idx=full_dataset.class_to_idx
    )

    val_dataset = GeoRock7ClassesDataset(
        root_dir=data_dir, 
        transform=transform_val,
        file_paths=[full_dataset.file_paths[i] for i in val_ds_raw.indices],
        labels=[full_dataset.labels[i] for i in val_ds_raw.indices],
        class_to_idx=full_dataset.class_to_idx
    )

    train_labels = train_dataset.labels
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    sample_weights = np.array([class_weights[t] for t in train_labels])
    
    # WeightedRandomSampler permet de tirer plus souvent les images des classes minoritaires
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(), 
        num_samples=len(sample_weights), 
        replacement=True
    )
    
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, sampler=sampler),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    }
    
    return dataloaders, full_dataset.class_to_idx, val_dataset

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=30, device='cpu', patience=7):
    """
    Boucle principale d'entraînement du modèle.
    Intègre un mécanisme de sauvegarde directe sur le disque (évitant les Out-Of-Memory)
    et de l'Early Stopping pour arrêter l'entraînement si la validation ne s'améliore plus.
    """
    best_acc = 0.0
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
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
            epoch_acc = running_corrects.double().item() / len(dataloaders[phase].dataset)
            
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc)
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc)
                scheduler.step(epoch_loss)
                
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    epochs_no_improve = 0
                    print('--> New best model saved to disk!')
                    torch.save(model.state_dict(), "best_georock_model_7classes.pth")
                    with open("best_model_arch.txt", "w") as f:
                        f.write("resnet50")
                else:
                    epochs_no_improve += 1
                    
        print()
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.\n")
            break
            
    print(f'Training Complete. Best Validation Accuracy: {best_acc:.4f}')
    return model, history

def plot_history(history, save_dir):
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_acc'], 'b-', label='Training Acc')
    plt.plot(epochs, history['val_acc'], 'r-', label='Validation Acc')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves_resnet.png"))
    plt.close()

def plot_confusion_matrix(labels, preds, target_names, save_dir):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix - ResNet50')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix_resnet.png"))
    plt.close()

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    data_dir = os.path.join("data", "Dataset")
    dataloaders, class_to_idx, val_dataset = get_dataloaders(data_dir, batch_size=8)
    
    num_classes = len(class_to_idx)
    
    print("Loading ResNet50 (Extreme Fine-Tuning)...")
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    
    for param in model.parameters():
        param.requires_grad = True
        
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )
    
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    # Réduit dynamiquement le taux d'apprentissage (learning rate) si l'apprentissage stagne
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    save_dir = "training_results"
    os.makedirs(save_dir, exist_ok=True)
    
    print("Starting Extreme Training...")
    best_model, history = train_model(
        model, 
        dataloaders, 
        criterion, 
        optimizer, 
        scheduler,
        num_epochs=40, 
        device=device,
        patience=7
    )
    
    plot_history(history, save_dir)
    
    # Load the best model from disk for final evaluation
    model.load_state_dict(torch.load("best_georock_model_7classes.pth", map_location=device))
    best_model = model
        
    print("Evaluating on Validation Set...")
    best_model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloaders['val']:
            inputs = inputs.to(device)
            outputs = best_model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    target_names = [idx_to_class[i] for i in range(num_classes)]
    
    plot_confusion_matrix(all_labels, all_preds, target_names, save_dir)
    
    report = classification_report(all_labels, all_preds, target_names=target_names)
    with open(os.path.join(save_dir, "classification_report_resnet.txt"), "w") as f:
        f.write(report)
        
    print("\nFinal classification report and graphs generated.")

if __name__ == '__main__':
    main()
