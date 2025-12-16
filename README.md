# 🎭 Tweet Emotion Classifier via Transformer

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)

Ce projet implémente un modèle d'IA basé sur l'architecture **Transformer** (construit "from scratch") pour classer les émotions dans des tweets. Le modèle est capable de distinguer 6 émotions principales : Joie, Tristesse, Colère, Peur, Amour et Surprise.

## 📄 Description du Projet

L'objectif est de démontrer l'efficacité de l'architecture Transformer (Self-Attention) pour le traitement du langage naturel (NLP) sur des textes courts et informels (tweets). Contrairement à l'utilisation de modèles pré-entraînés (comme BERT), ce projet construit et entraîne un encodeur Transformer léger entièrement personnalisé avec TensorFlow/Keras.

### Caractéristiques principales :
* **Nettoyage de texte** : Suppression des URLs, mentions (@user) et caractères spéciaux via Regex.
* **Architecture** : Couche d'Embedding personnalisée (Token + Position) et Bloc Transformer (Multi-Head Attention).
* **Entraînement** : Optimisation avec Adam, gestion du sur-apprentissage via Dropout et EarlyStopping.
* **Sauvegarde** : Checkpoint automatique du meilleur modèle sur Google Drive.

## 📊 Dataset

Le modèle est entraîné sur le dataset **[dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion)** disponible sur Hugging Face.

* **Taille** : ~20 000 tweets (Train/Val/Test).
* **Classes** :
    * 0: Sadness (Tristesse)
    * 1: Joy (Joie)
    * 2: Love (Amour)
    * 3: Anger (Colère)
    * 4: Fear (Peur)
    * 5: Surprise

## 🧠 Architecture du Modèle

Le modèle n'utilise pas de RNN (LSTM/GRU) mais repose entièrement sur l'attention :

1.  **Input Layer** : Séquences de longueur fixe (100 tokens).
2.  **Token & Position Embedding** : Combine le sens du mot et sa position dans la phrase.
3.  **Transformer Block** :
    * Multi-Head Attention (2 têtes).
    * Réseau Feed-Forward (Dense).
    * Normalisation (LayerNorm) et connexions résiduelles.
4.  **Global Average Pooling** : Réduction de la dimensionnalité.
5.  **Output Layer** : Dense + Softmax (6 neurones).

## 📈 Performances et Résultats

* **Accuracy (Test)** : ~89%.
* **Observations** : Le modèle distingue très bien les émotions à forte valence (Joie vs Colère) mais peut présenter des confusions sur des nuances sémantiques proches (ex: Tristesse vs Peur).


`![Confusion Matrix](./Matrice_confusion.png)`

## 🛠️ Installation et Utilisation

### Prérequis
* Python 3.x
* TensorFlow / Keras
* Datasets (Hugging Face)
* Pandas, NumPy, Scikit-learn, Matplotlib

### Installation
```bash
pip install tensorflow datasets pandas numpy scikit-learn matplotlib seaborn
