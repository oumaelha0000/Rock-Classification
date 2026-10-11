# Rapport de Projet : GeoRock AI - Classification de Roches par Deep Learning

## 1. Introduction & Problématique
L'identification géologique précise des roches est une compétence qui requiert souvent des années d'expertise. Les géologues, étudiants et amateurs manquent fréquemment d'outils d'assistance instantanée sur le terrain. L'objectif du projet **GeoRock AI** est de développer une solution d'Intelligence Artificielle capable de classifier des échantillons de roches à partir d'une simple photographie.

Dans un souci de transparence scientifique, ce projet ne s'arrête pas à la simple prédiction, mais intègre une interface web ainsi qu'un module d'**Explicabilité (XAI)** pour comprendre le raisonnement du modèle.

---

## 2. Le Jeu de Données (Dataset)
Le projet initial comportait 3 classes de roches. Pour obtenir un système robuste et utile, nous avons étendu la base de données à **7 classes distinctes** en fusionnant notre jeu de données avec des bases de données ouvertes (Kaggle).

Les 7 classes finales sont : 
- **Basalte, Charbon (Coal), Granite, Calcaire (Limestone), Marbre, Quartzite, Grès (Sandstone)**.

Afin de contrer le léger déséquilibre des classes et d'améliorer la généralisation du modèle, nous avons implémenté des techniques avancées lors du chargement des données :
* **WeightedRandomSampler** : Pénalise les classes majoritaires et donne plus de poids aux classes minoritaires.
* **Data Augmentation Extrême** : Rotations aléatoires, inversions horizontales/verticales, et modification dynamique de la colorimétrie (ColorJitter) pour simuler différentes conditions d'éclairage sur le terrain.

---

## 3. Modélisation et Entraînement

### Comparaison des Modèles
Nous avons initialement testé et comparé plusieurs architectures de réseaux de neurones convolutifs (CNN) pré-entraînés :
1. **CNN Personnalisé** (Modèle de base, performances limitées)
2. **ResNet18** (Bon compromis vitesse/précision)
3. **MobileNetV2** (Très rapide, a atteint ~73.2% de précision)

![Comparaison des Modèles](/C:/Users/elbad/.gemini/antigravity-ide/brain/d26b1bd8-dd4e-49b9-8939-8955c9642e38/model_comparison.png)

### Le Choix Final : ResNet50
Pour maximiser la précision, nous avons finalement opté pour une architecture **ResNet50** très profonde. Les dernières couches (Classifieur) ont été remplacées par des couches denses intégrant du Dropout (0.5) pour éviter le surapprentissage (overfitting).

L'entraînement complet du ResNet50 demandant d'énormes ressources de calcul, le code a été modularisé et déporté sur les GPU de **Google Colab**, permettant une exécution rapide tout en soulageant le processeur (CPU) local de l'ordinateur.

### Évaluation du Modèle
La matrice de confusion ci-dessous montre la capacité du modèle à distinguer efficacement les roches sédimentaires des roches métamorphiques et ignées.

![Matrice de Confusion](/C:/Users/elbad/.gemini/antigravity-ide/brain/d26b1bd8-dd4e-49b9-8939-8955c9642e38/confusion_matrix.png)

---

## 4. L'Application Web & L'Interface Utilisateur
Pour rendre le modèle accessible, nous avons développé une application web moderne utilisant **Flask** (Python) en backend et **HTML/CSS/JS Vanilla** en frontend.

L'interface a été conçue pour être "Premium" avec un design sombre (Dark Mode), des animations fluides, et une galerie intégrée pour des tests rapides.

![Aperçu de l'Interface](/C:/Users/elbad/.gemini/antigravity-ide/brain/d26b1bd8-dd4e-49b9-8939-8955c9642e38/georock_preview_1782636492766.webp)

---

## 5. Explicabilité de l'IA (Grad-CAM)
C'est la fonctionnalité phare (Killer Feature) du projet. L'Intelligence Artificielle étant souvent perçue comme une boîte noire, nous avons intégré l'algorithme **Grad-CAM**.

Lorsqu'un utilisateur soumet une image, le serveur ne renvoie pas qu'une simple prédiction. Il génère dynamiquement une "carte de chaleur" (heatmap) visuelle superposée à l'image d'origine. Les zones rouges montrent les pixels exacts que le réseau de neurones a pris en compte pour prendre sa décision. 
Cela permet au géologue de vérifier que l'IA détecte bien la texture de la roche, et non un élément du décor, assurant ainsi une confiance totale dans le système.

---

## 6. Conclusion
GeoRock AI démontre avec succès comment le Deep Learning (ResNet50) couplé à une interface utilisateur intuitive et à de l'Intelligence Artificielle Explicable (XAI) peut fournir un outil de terrain robuste et digne de confiance pour les Sciences de la Terre. Le code a été optimisé, documenté et préparé pour d'éventuelles évolutions futures.
