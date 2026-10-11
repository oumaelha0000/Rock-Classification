"""
Module de gestion du Dataset Géologique.
Permet de charger dynamiquement les images des roches, de les associer à 
leurs classes respectives et d'appliquer des transformations (Data Augmentation).
"""
import os
from torch.utils.data import Dataset
from PIL import Image

class GeoRock7ClassesDataset(Dataset):
    """
    Dataset PyTorch personnalisé pour les 7 classes de roches.
    
    Cette classe parcourt automatiquement l'arborescence des dossiers
    (Super-classe -> Sous-classe -> Image) pour construire le jeu de données
    sans nécessiter de fichier CSV d'annotations.
    
    Structure attendue du dossier :
    root_dir/
    ├── Igneous/
    │   ├── Basalt/
    │   └── Granite/
    ├── Metamorphic/
    ...
    """
    def __init__(self, root_dir, transform=None, file_paths=None, labels=None, class_to_idx=None):
        """
        Initialise le Dataset. Si les chemins et labels ne sont pas fournis (comme lors
        de l'instanciation principale), la fonction explore les dossiers pour les générer.
        """
        self.root_dir = root_dir
        self.transform = transform
        
        if file_paths is None or labels is None or class_to_idx is None:
            # Nous définissons explicitement les 7 classes cibles
            self.classes = sorted(['Basalt', 'Coal', 'Granite', 'Limestone', 'Marble', 'Quartzite', 'Sandstone'])
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
            
            self.file_paths = []
            self.labels = []
            
            # Parcours des super-classes (Igneous, Metamorphic, Sedimentary)
            for superclass_name in os.listdir(root_dir):
                superclass_dir = os.path.join(root_dir, superclass_name)
                if not os.path.isdir(superclass_dir):
                    continue
                
                # Parcours des sous-types (Basalt, Granite...)
                for subtype in os.listdir(superclass_dir):
                    subtype_dir = os.path.join(superclass_dir, subtype)
                    if not os.path.isdir(subtype_dir):
                        continue
                    
                    if subtype not in self.class_to_idx:
                        continue # Ignore les dossiers qui ne font pas partie des 7 classes
                        
                    for img_name in os.listdir(subtype_dir):
                        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                            self.file_paths.append(os.path.join(subtype_dir, img_name))
                            self.labels.append(self.class_to_idx[subtype])
        else:
            self.file_paths = file_paths
            self.labels = labels
            self.class_to_idx = class_to_idx
            self.classes = {v: k for k, v in class_to_idx.items()}

    def __len__(self):
        """Retourne le nombre total d'images dans le dataset."""
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        Charge une image spécifique depuis le disque dur, la convertit en RGB,
        et lui applique les éventuelles transformations (Data Augmentation).
        """
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label
