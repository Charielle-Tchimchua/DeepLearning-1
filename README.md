# MNIST Classification avec Keras

Un projet simple de classification d'images MNIST utilisant Keras et TensorFlow.

## Description

Ce projet implémente un modèle de réseau de neurones pour classifier les chiffres manuscrits du jeu de données MNIST. Le modèle utilise une architecture fully-connected avec :
- Une couche dense de 512 neurones avec activation ReLU
- Une couche de dropout (0.2) pour la régularisation
- Une couche de sortie avec 10 neurones (pour les 10 chiffres) et activation softmax

## Structure du projet

```
mnist_keras_project/
├── mnist_model.h5          # Modèle pré-entraîné
├── train_model.py          # Script d'entraînement
├── README.md               # Ce fichier
└── venv/                   # Environnement virtuel Python
```

## Prérequis

- Python 3.12+
- TensorFlow 2.x
- Keras
- NumPy

## Installation

1. Clone ce dépôt (si applicable)
2. Crée un environnement virtuel :
   ```bash
   python -m venv venv
   ```

3. Active l'environnement virtuel :
   - Sur Linux/Mac:
     ```bash
     source venv/bin/activate
     ```
   - Sur Windows:
     ```bash
     venv\Scripts\activate
     ```

4. Installe les dépendances :
   ```bash
   pip install tensorflow numpy
   ```

## Utilisation

### Entraîner le modèle

Pour entraîner le modèle à partir de zéro :

```bash
python train_model.py
```

Le script va :
1. Charger le jeu de données MNIST
2. Normaliser et redimensionner les images
3. Construire et compiler le modèle
4. Entraîner le modèle pendant 5 epochs
5. Évaluer le modèle sur les données de test
6. Sauvegarder le modèle sous `mnist_model.h5`

### Utiliser le modèle pré-entraîné

Le modèle est déjà entraîné et sauvegardé dans `mnist_model.h5`. Vous pouvez le charger et l'utiliser directement :

```python
from tensorflow import keras

# Charger le modèle
model = keras.models.load_model("mnist_model.h5")

# Faire des prédictions
# (assurez-vous que vos données sont normalisées et redimensionnées comme dans train_model.py)
predictions = model.predict(vos_donnees)
```

## Détails du modèle

- **Architecture** : Réseau fully-connected (Dense)
- **Couches** :
  - Dense(512, activation='relu')
  - Dropout(0.2)
  - Dense(10, activation='softmax')
- **Optimiseur** : Adam
- **Fonction de perte** : sparse_categorical_crossentropy
- **Métriques** : accuracy
- **Batch size** : 128
- **Épochs** : 5
- **Validation split** : 10%

## Performances

Le modèle atteint généralement une précision d'environ 98% sur les données de test MNIST après 5 epochs d'entraînement.

## Notes

- Le script désactive l'utilisation du GPU (`CUDA_VISIBLE_DEVICES="-1"`) pour éviter les problèmes de configuration
- Les images sont normalisées (valeurs entre 0 et 1) et redimensionnées en vecteurs de 784 éléments (28x28 pixels)
- Le modèle utilise un dropout de 20% pour prévenir le surapprentissage

## Auteur

line_charie

## Licence

Ce projet est sous licence MIT (ou autre licence de votre choix - à spécifier).