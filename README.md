# 🍎 Détection de Fruits avec YOLOv8

Projet de **Deep Learning pour la détection d’objets** dans des images, réalisé dans le cadre du cours **Deep Learning for Computer Vision** du **Master SISE – Université Lumière Lyon 2**.

Auteur : **Aya Mecheri**  
Année universitaire : **2025 / 2026**

---

# 📌 Objectif du projet

L’objectif de ce projet est de développer un modèle capable de **détecter et classifier automatiquement plusieurs fruits dans une image** à l’aide d’un réseau neuronal profond.

Contrairement à une simple classification d’image, la détection d’objets consiste à :

- identifier la présence d’un objet
- déterminer sa classe
- localiser sa position dans l’image à l’aide d’une bounding box

Le modèle choisi pour cette tâche est **YOLOv8n (You Only Look Once)**, une architecture moderne de détection d’objets permettant d’obtenir des **prédictions rapides et précises**.

---

# 🧠 Architecture utilisée

Le modèle utilisé est **YOLOv8n**, la version la plus légère de YOLOv8 développée par **Ultralytics**.

L’architecture se compose de trois parties principales :

### Backbone

Le backbone extrait les caractéristiques visuelles de l’image (textures, formes, contours).  
YOLOv8 utilise une architecture inspirée de **CSPDarknet** avec des blocs **C2f** qui améliorent le flux de gradient et la stabilité de l’entraînement.

### Neck

La partie Neck combine les caractéristiques à différentes échelles grâce aux structures :

- FPN (Feature Pyramid Network)
- PAN (Path Aggregation Network)

Cela permet de détecter des objets de différentes tailles.

### Head

La Head du réseau prédit :

- les bounding boxes
- les classes
- les scores de confiance

pour chaque objet détecté.

---

# 🔁 Transfer Learning

Étant donné la taille relativement modeste du dataset, le modèle est entraîné en utilisant **le Transfer Learning**.

Le processus est le suivant :

1. Chargement d’un modèle pré-entraîné sur le dataset COCO.
2. Réutilisation des features visuelles déjà apprises.
3. Fine-tuning du modèle sur notre dataset spécifique de fruits.

Cette stratégie permet une **convergence plus rapide** et de **meilleures performances**.

---

# 📊 Dataset

Le dataset utilisé dans ce projet a été **entièrement créé manuellement**.

Les images ont été prises avec un **smartphone** dans différentes conditions :

- éclairage variable
- différents fonds
- plusieurs fruits dans une même image

### Caractéristiques du dataset

Nombre total d’images : **359**

Nombre de classes : **5**

Classes détectées :

| ID | Classe |
|----|------|
| 0 | Pomme |
| 1 | Banane |
| 2 | Kiwi |
| 3 | Citron |
| 4 | Mandarine |

Chaque image peut contenir **plusieurs fruits simultanément**.

---

# 🏷️ Annotation des données

Les annotations ont été réalisées avec l’outil :

**CVAT (Computer Vision Annotation Tool)**

Pour chaque image :

- une bounding box est dessinée autour du fruit
- la classe correspondante est attribuée

Les annotations sont exportées au **format YOLO** :

```
class_id x_center y_center width height
```

Toutes les coordonnées sont **normalisées entre 0 et 1**.

---

# 🔀 Répartition des données

Le dataset est divisé en trois sous-ensembles :

| Split | Images | Proportion |
|------|------|------|
| Train | 251 | 70 % |
| Validation | 53 | 15 % |
| Test | 55 | 15 % |

- **Train** : entraînement du modèle  
- **Validation** : réglage des hyperparamètres  
- **Test** : évaluation finale

---

# ⚙️ Hyperparamètres d’entraînement

Les principaux hyperparamètres utilisés sont :

```
epochs = 50
imgsz = 640
batch = 16
optimizer = AdamW
learning_rate = 0.01
patience = 15
```

### Justification

- **epochs = 50** : suffisant pour le fine-tuning  
- **imgsz = 640** : taille standard YOLO  
- **batch = 16** : adapté au GPU T4  
- **AdamW** : bonne convergence pour les petits datasets  
- **early stopping** pour éviter le surapprentissage

---

# 🔧 Data Augmentation

Pour améliorer la généralisation du modèle, plusieurs techniques d’augmentation sont utilisées :

- Mosaic augmentation
- Flip horizontal / vertical
- Rotation aléatoire
- Variation HSV (saturation et luminosité)

Ces techniques permettent d’augmenter artificiellement la diversité des données.

---

# 📈 Résultats expérimentaux

Les performances du modèle sont évaluées sur le **jeu de test** (55 images).

### Résultats globaux

| Métrique | Valeur |
|------|------|
| mAP50 | 93.1 % |
| mAP50-95 | 59.3 % |
| Precision | 91.1 % |
| Recall | 92.1 % |

Le modèle obtient une **très bonne précision globale**.

---

# 📊 Résultats par classe

| Classe | AP50 |
|------|------|
| Pomme | 99.5 % |
| Mandarine | 99.1 % |
| Kiwi | 97.4 % |
| Citron | 95.6 % |
| Banane | 74.3 % |

Les **bananes** sont plus difficiles à détecter à cause de leur forme allongée.

---

# 🧪 Analyse qualitative

### Cas réussis

Le modèle est capable de détecter **plusieurs fruits dans une même image** avec des scores de confiance élevés.

### Erreurs observées

Trois types d’erreurs apparaissent :

- double détection du même fruit
- détection avec faible confiance
- chevauchement de bounding boxes lorsque les fruits sont proches

---

# 📉 Absence de surapprentissage

Les courbes d’entraînement montrent :

- une diminution conjointe des losses train et validation
- aucune divergence importante

Cela indique une **bonne capacité de généralisation**.

---

# ⚠️ Limites du projet

- dataset relativement petit (359 images)
- certaines classes plus difficiles (banane)
- modèle YOLOv8n très léger

---

# 🚀 Améliorations possibles

Plusieurs pistes d’amélioration :

- augmenter la taille du dataset
- équilibrer les classes
- tester YOLOv8s ou YOLOv8m
- ajuster les hyperparamètres

---

# 📂 Structure du projet

```
fruit-detection-yolov8
│
├── dataset
│   ├── images
│   ├── labels
│   └── data.yaml
│
├── notebooks
├── report
│   └── deep_learning_rapport.pdf
│
└── README.md
```

---

# 🧑‍💻 Exemple d’entraînement

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="dataset/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16
)
```

---

# 📚 Références

- Ultralytics YOLOv8  
- COCO Dataset  
- PyTorch  
- CVAT

---

# 🎓 Conclusion

Ce projet a permis de mettre en pratique plusieurs concepts clés du **Deep Learning appliqué à la vision par ordinateur** :

- détection d’objets
- transfer learning
- création et annotation d’un dataset
- entraînement et évaluation d’un modèle

Malgré un dataset relativement limité, le modèle obtient des **résultats très satisfaisants avec un mAP50 de 93.1 %**.