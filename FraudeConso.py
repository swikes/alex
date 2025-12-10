# -*- coding: utf-8 -*-
"""
Étape 1 - Chargement des données et EDA basique
Projet : Détection de fraude à partir de l'historique de consommation
"""

# 1. Import des librairies de base
import pandas as pd
import numpy as np

# (Optionnel pour plus tard : visualisation)
import matplotlib.pyplot as plt

# 2. Options d'affichage pour mieux lire les tableaux
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# 3. Chemin du fichier à adapter à ton environnement
# Exemple : r"C:\Users\TonNom\Documents\Dataset_ProjetML.xlsx"
file_path = r"C:\Users\adamo\OneDrive\Documentos\IUA\Dataset_ProjetML.xlsx" # 

# 4. Chargement du fichier Excel
df = pd.read_excel(file_path, sheet_name="Sheet1")

print("✅ Données chargées avec succès")
print("-" * 80)

# 5. Taille du dataset
print("Shape (lignes, colonnes) : ", df.shape)
print("-" * 80)

# 6. Aperçu des premières lignes
print("Aperçu des 5 premières lignes :")
print(df.head())
print("-" * 80)

# 7. Info sur les types de variables
print("Info sur les types de variables :")
print(df.info())
print("-" * 80)

# 8. Vérification des valeurs manquantes
print("Nombre de valeurs manquantes par colonne (top 15) :")
print(df.isna().sum().sort_values(ascending=False).head(15))
print("-" * 80)

# 9. Distribution de la variable cible (FRAUDEUR)
if "FRAUDEUR" in df.columns:
    print("Distribution de la variable FRAUDEUR :")
    print(df["FRAUDEUR"].value_counts())
    print("\nDistribution relative (en %) :")
    print(round(df["FRAUDEUR"].value_counts(normalize=True) * 100, 2))
else:
    print("⚠️ Attention : la colonne 'FRAUDEUR' n'existe pas dans le dataset.")

print("-" * 80)

# 10. Quelques statistiques descriptives sur les variables numériques
print("Statistiques descriptives des variables numériques :")
print(df.describe())
print("-" * 80)

# ================================
# ÉTAPE 2.1 - Suppression des doublons
# ================================

print("Nombre de doublons AVANT suppression :", df.duplicated().sum())

df = df.drop_duplicates()

print("Nombre de doublons APRÈS suppression :", df.duplicated().sum())
print("Nouvelle taille du dataset :", df.shape)
print("-" * 80)

# ================================
# ÉTAPE 2.2 - Nettoyage de la variable cible
# ================================

# Vérification des valeurs uniques
print("Valeurs uniques de la variable FRAUDEUR :")
print(df["FRAUDEUR"].unique())

# Conversion en int si nécessaire
df["FRAUDEUR"] = df["FRAUDEUR"].astype(int)

print("Type de la variable FRAUDEUR après correction :", df["FRAUDEUR"].dtype)
print("-" * 80)

# ================================
# ÉTAPE 2.3 - Gestion des valeurs manquantes
# ================================

# Séparation des colonnes numériques et catégorielles
cols_num = df.select_dtypes(include=["int64", "float64"]).columns
cols_cat = df.select_dtypes(include=["object"]).columns

print("Colonnes numériques :", cols_num)
print("Colonnes catégorielles :", cols_cat)
print("-" * 80)

# Remplacement des valeurs manquantes :
# - numériques → médiane
# - catégorielles → modalité la plus fréquente

for col in cols_num:
    df[col] = df[col].fillna(df[col].median())

for col in cols_cat:
    df[col] = df[col].fillna(df[col].mode()[0])

# Vérification finale
print("Valeurs manquantes restantes :")
print(df.isna().sum().sort_values(ascending=False).head(10))
print("-" * 80)

# ================================
# ÉTAPE 2.4 - Suppression des colonnes non utiles
# ================================

colonnes_a_supprimer = [
    "REFERENCE_CONTRAT", 
    "PERIODE_FACTURATION"
]

# Suppression seulement si elles existent
df = df.drop(columns=[col for col in colonnes_a_supprimer if col in df.columns])

print("Colonnes restantes après suppression :")
print(df.columns)
print("-" * 80)

# ================================
# ÉTAPE 2.5 - Encodage des variables catégorielles
# ================================

df_encoded = pd.get_dummies(df, drop_first=True)

print("Dataset après encodage :")
print(df_encoded.shape)
print(df_encoded.head())
print("-" * 80)

# ================================
# ÉTAPE 3 - Préparation des données pour les modèles
# ================================

from sklearn.preprocessing import StandardScaler

# 3.0 - On garde UNIQUEMENT les colonnes numériques
df_num = df_encoded.select_dtypes(include=[np.number]).copy()

print("Shape df_encoded :", df_encoded.shape)
print("Shape df_num (numériques seulement) :", df_num.shape)
print("-" * 80)

# Vérif de contrôle : il ne doit plus rester de colonnes non numériques
print("Types restants dans df_num :")
print(df_num.dtypes.value_counts())
print("-" * 80)

# 3.1 - Séparation X / y
y = df_num["FRAUDEUR"].astype(int)
X = df_num.drop("FRAUDEUR", axis=1)

print("Dimensions de X :", X.shape)
print("Dimensions de y :", y.shape)
print("-" * 80)

# 3.2 - Normalisation des variables explicatives
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("✅ Normalisation terminée")
print("Shape de X_scaled :", X_scaled.shape)
print("-" * 80)

# 3.3 - Vérification du déséquilibre des classes
unique, counts = np.unique(y, return_counts=True)
print("Répartition des classes :", dict(zip(unique, counts)))
print("Pourcentage de fraudeurs :", round(counts[1] / sum(counts) * 100, 2), "%")
print("-" * 80)

"""
# ================================
# ÉTAPE 3.4 - Rééquilibrage avec SMOTE
# ================================

from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

print("✅ Données après SMOTE :")
print("X :", X_resampled.shape)
print("y :", y_resampled.shape)
print("-" * 80)
"""
# ================================
# ÉTAPE 4.1 - Séparation Train / Test
# ================================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, 
    y,
    test_size=0.2,          # 80% train, 20% test
    random_state=42,
    stratify=y              # très important en cas de classes déséquilibrées
)

print("Taille X_train :", X_train.shape)
print("Taille X_test  :", X_test.shape)
print("Taille y_train :", y_train.shape)
print("Taille y_test  :", y_test.shape)
print("-" * 80)

# ================================
# ÉTAPE 4.2 - Rééquilibrage du train avec SMOTE
# ================================

from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("Après SMOTE :")
print("X_train_smote :", X_train_smote.shape)
print("y_train_smote :", y_train_smote.shape)

# Vérification de la nouvelle répartition des classes
unique_smote, counts_smote = np.unique(y_train_smote, return_counts=True)
print("Répartition des classes dans y_train_smote :", dict(zip(unique_smote, counts_smote)))
print("-" * 80)

# ================================
# ÉTAPE 5.1 - Régression Logistique
# ================================

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Entraînement
log_model = LogisticRegression(max_iter=1000, n_jobs=-1)
log_model.fit(X_train_smote, y_train_smote)

# Prédictions sur le jeu de test
y_pred_log = log_model.predict(X_test)

# Évaluation
print("✅ MODÈLE : RÉGRESSION LOGISTIQUE")
print("Accuracy :", accuracy_score(y_test, y_pred_log))
print("Matrice de confusion :\n", confusion_matrix(y_test, y_pred_log))
print("\nRapport de classification :\n", classification_report(y_test, y_pred_log))
print("-" * 80)

# ================================
# ÉTAPE 5.2 - Random Forest
# ================================

from sklearn.ensemble import RandomForestClassifier

# Entraînement
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_smote, y_train_smote)

# Prédictions sur le jeu de test
y_pred_rf = rf_model.predict(X_test)

# Évaluation
print("✅ MODÈLE : RANDOM FOREST")
print("Accuracy :", accuracy_score(y_test, y_pred_rf))
print("Matrice de confusion :\n", confusion_matrix(y_test, y_pred_rf))
print("\nRapport de classification :\n", classification_report(y_test, y_pred_rf))
print("-" * 80)

# ================================
# ÉTAPE 6.1 - Probabilités des modèles
# ================================

y_proba_log = log_model.predict_proba(X_test)[:, 1]
y_proba_rf  = rf_model.predict_proba(X_test)[:, 1]

# ================================
# ÉTAPE 6.2 - Courbes ROC et AUC
# ================================

from sklearn.metrics import roc_curve, roc_auc_score

# Calcul AUC
auc_log = roc_auc_score(y_test, y_proba_log)
auc_rf  = roc_auc_score(y_test, y_proba_rf)

print("AUC Régression Logistique :", auc_log)
print("AUC Random Forest :", auc_rf)

# Courbes ROC
fpr_log, tpr_log, _ = roc_curve(y_test, y_proba_log)
fpr_rf,  tpr_rf,  _ = roc_curve(y_test, y_proba_rf)

plt.figure()
plt.plot(fpr_log, tpr_log, label="Logistique (AUC = {:.3f})".format(auc_log))
plt.plot(fpr_rf,  tpr_rf,  label="Random Forest (AUC = {:.3f})".format(auc_rf))
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("Taux de faux positifs")
plt.ylabel("Taux de vrais positifs")
plt.title("Courbes ROC - Détection de fraude")
plt.legend()
plt.grid()
plt.show()

# ================================
# ÉTAPE 7.1 - Importance des variables du Random Forest
# ================================

import pandas as pd
import numpy as np

feature_names = X.columns

importances = rf_model.feature_importances_

df_importances = pd.DataFrame({
    "Variable": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("Top 15 variables les plus importantes :")
print(df_importances.head(15))

# ================================
# ÉTAPE 7.2 - Visualisation des 10 variables les plus importantes
# ================================

top10 = df_importances.head(10)

plt.figure()
plt.barh(top10["Variable"], top10["Importance"])
plt.gca().invert_yaxis()
plt.xlabel("Importance")
plt.title("Top 10 des variables les plus importantes - Random Forest")
plt.grid()
plt.show()

# ================================
# EXTRA - Visualisation des images impoprtantes
# ================================


import matplotlib.pyplot as plt
import numpy as np

## Matrice de confusion - Matrice de confusion - Logistique Regression
cm_log = np.array([[1822, 266],
                   [1348, 1007]])

plt.figure()
plt.imshow(cm_log)
plt.title("Matrice de confusion - Régression Logistique")
plt.xlabel("Prédit")
plt.ylabel("Réel")
plt.colorbar()
plt.tight_layout()
plt.show()

## Matrice de confusion - Random Forest
cm_rf = np.array([[1752, 336],
                  [487, 1868]])

plt.figure()
plt.imshow(cm_rf)
plt.title("Matrice de confusion - Random Forest")
plt.xlabel("Prédit")
plt.ylabel("Réel")
plt.colorbar()
plt.tight_layout()
plt.show()

## Comparaison des modèles
models = ["Logistique", "Random Forest"]
accuracy = [0.637, 0.815]
recall = [0.43, 0.79]
f1 = [0.56, 0.82]
auc = [0.695, 0.895]

x = np.arange(len(models))

plt.figure()
plt.bar(x - 0.3, accuracy, width=0.2)
plt.bar(x - 0.1, recall, width=0.2)
plt.bar(x + 0.1, f1, width=0.2)
plt.bar(x + 0.3, auc, width=0.2)

plt.xticks(x, models)
plt.xlabel("Modèles")
plt.ylabel("Score")
plt.title("Comparaison des performances des modèles")
plt.legend(["Accuracy", "Recall Fraude", "F1 Fraude", "AUC"])
plt.tight_layout()
plt.show()

## Top 10 variables importantes

features = ["MOYENNE_12m", "ANNEE_ANCIENNETE", "CONSO_SAISONNIERE",
            "MOY_MENSUELLE", "MONTANT_FACTURE", "PUISSANCE_SOUSCRITE",
            "CONSO_KWH", "RATIO_CONSO_PS", "TAUX_VARIATION", "RATIO_FACTURE_CONSO"]

importances = [0.155660, 0.125023, 0.094815, 0.091084, 0.090955,
               0.081142, 0.068287, 0.062095, 0.051642, 0.050070]

plt.figure()
plt.barh(features, importances)
plt.xlabel("Importance")
plt.title("Top 10 des variables les plus importantes - Random Forest")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

## Distribution des classes SMOTE

classes = ["Non fraudeur", "Fraudeur"]
values = [9420, 9420]

plt.figure()
plt.bar(classes, values)
plt.xlabel("Classe")
plt.ylabel("Effectif")
plt.title("Distribution des classes après SMOTE")
plt.tight_layout()
plt.show()

## Courbe ROC

from sklearn.metrics import roc_curve, roc_auc_score

# Probabilités
y_proba_log = log_model.predict_proba(X_test)[:, 1]
y_proba_rf  = rf_model.predict_proba(X_test)[:, 1]

# AUC
auc_log = roc_auc_score(y_test, y_proba_log)
auc_rf  = roc_auc_score(y_test, y_proba_rf)

# ROC
fpr_log, tpr_log, _ = roc_curve(y_test, y_proba_log)
fpr_rf,  tpr_rf,  _ = roc_curve(y_test, y_proba_rf)

plt.figure()
plt.plot(fpr_log, tpr_log, label="Logistique (AUC = {:.3f})".format(auc_log))
plt.plot(fpr_rf,  tpr_rf,  label="Random Forest (AUC = {:.3f})".format(auc_rf))
plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("Taux de faux positifs")
plt.ylabel("Taux de vrais positifs")
plt.title("Courbes ROC - Détection de fraude")
plt.legend()
plt.tight_layout()
plt.show()
