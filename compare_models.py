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
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
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
    
    # Weights for class imbalance
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    sample_weights = np.array([class_weights[label] for label in train_labels])
    sampler = WeightedRandomSampler(weights=torch.from_numpy(sample_weights).double(), num_samples=len(sample_weights), replacement=True)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, full_dataset.class_to_idx

def train_model(model, dataloaders, criterion, optimizer, num_epochs=5, device='cpu'):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        
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
            
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
    model.load_state_dict(best_model_wts)
    return model, history, best_acc

def create_model(model_name, num_classes):
    if model_name == 'CustomCNN':
        return CustomCNN(num_classes=num_classes)
    elif model_name == 'ResNet18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_name == 'MobileNetV2':
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    elif model_name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        return model
    else:
        raise ValueError(f"Unknown model {model_name}")

def plot_comparison(all_histories, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(14, 6))
    
    # Plot Validation Accuracy
    plt.subplot(1, 2, 1)
    for model_name, history in all_histories.items():
        plt.plot(range(1, len(history['val_acc']) + 1), history['val_acc'], marker='o', label=model_name)
    plt.title('Validation Accuracy Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot Validation Loss
    plt.subplot(1, 2, 2)
    for model_name, history in all_histories.items():
        plt.plot(range(1, len(history['val_loss']) + 1), history['val_loss'], marker='o', label=model_name)
    plt.title('Validation Loss Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'))
    plt.close()

def plot_final_accuracies(best_accs, output_dir):
    plt.figure(figsize=(8, 6))
    models_names = list(best_accs.keys())
    accs = [acc.item() * 100 for acc in best_accs.values()]
    
    sns.barplot(x=models_names, y=accs, palette='viridis')
    plt.title('Final Validation Accuracy by Model (%)')
    plt.ylabel('Accuracy (%)')
    for i, v in enumerate(accs):
        plt.text(i, v + 1, f"{v:.1f}%", ha='center', fontweight='bold')
    plt.ylim(0, 110)
    plt.savefig(os.path.join(output_dir, 'final_accuracies.png'))
    plt.close()

def main():
    data_dir = 'data/Dataset'
    output_dir = 'model_comparison_results'
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_loader, val_loader, class_to_idx = get_dataloaders(data_dir, batch_size=16)
    dataloaders = {'train': train_loader, 'val': val_loader}
    num_classes = len(class_to_idx)
    
    print(f"Training on 7 classes: {class_to_idx}")
    
    models_to_test = ['CustomCNN', 'ResNet18', 'MobileNetV2']
    all_histories = {}
    best_accs = {}
    
    global_best_acc = 0.0
    global_best_model_name = ""
    
    # We train for a short number of epochs (e.g. 5) for quick comparison on CPU
    NUM_EPOCHS = 5
    
    for model_name in models_to_test:
        print(f"\n{'='*40}")
        print(f"Training {model_name}")
        print(f"{'='*40}")
        
        model = create_model(model_name, num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        trained_model, history, best_acc = train_model(
            model, dataloaders, criterion, optimizer, num_epochs=NUM_EPOCHS, device=device
        )
        
        all_histories[model_name] = history
        best_accs[model_name] = best_acc
        
        if best_acc > global_best_acc:
            global_best_acc = best_acc
            global_best_model_name = model_name
            torch.save(trained_model.state_dict(), 'best_georock_model_7classes.pth')
            with open('best_model_arch.txt', 'w') as f:
                f.write(model_name)
            print(f"-> New global best model: {model_name} ({best_acc:.4f}) saved.")

    print("\nTraining complete. Generating comparison graphs...")
    plot_comparison(all_histories, output_dir)
    plot_final_accuracies(best_accs, output_dir)
    print(f"Best overall model: {global_best_model_name} with {global_best_acc:.4f} accuracy.")

if __name__ == '__main__':
    main()
