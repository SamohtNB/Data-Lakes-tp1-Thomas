import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import tqdm
import joblib
from collections import OrderedDict


def preprocess_data(data_file, output_dir):
    """
    Exercice : Fonction pour prétraiter les données brutes et les préparer pour l'entraînement de modèles.

    Objectifs :
    1. Charger les données brutes à partir d’un fichier CSV.
    2. Nettoyer les données (par ex. : supprimer les valeurs manquantes).
    3. Encoder les labels catégoriels (colonne `family_accession`) en entiers.
    4. Diviser les données en ensembles d’entraînement, de validation et de test selon une logique définie.
    5. Sauvegarder les ensembles prétraités et des métadonnées utiles.

    Indices :
    - Utilisez `LabelEncoder` pour encoder les catégories.
    - Utilisez `train_test_split` pour diviser les indices des données.
    - Utilisez `to_csv` pour sauvegarder les fichiers prétraités.
    - Calculez les poids de classes en utilisant les comptes des classes.
    """

    # Step 1: Load the data
    print('Loading Data')
    data = pd.read_csv(data_file)

    # Step 2: Handle missing values
    data = data.dropna()

    # Step 3: Encode the 'family_accession' to numeric labels
    label_encoder = LabelEncoder()
    data['family_accession'] = label_encoder.fit_transform(data['family_accession'])

    # Save the label encoder
    joblib.dump(label_encoder, f"{output_dir}/label_encoder.pkl")

    # Save the label mapping to a text file
    with open(f"{output_dir}/label_mapping.txt", "w") as f:
        for i, label in enumerate(label_encoder.classes_):
            f.write(f"{i}: {label}\n")

    # Step 4: Distribute data
    # For each unique class:
    # - If count == 1: go to test set
    # - If count == 2: 1 to dev, 1 to test
    # - If count == 3: 1 to train, 1 to dev, 1 to test
    # - Else: stratified split (train/dev/test)
    
    print("Distributing data")
    test_set = []
    dev_set = []
    train_set = []
    
    for cls in tqdm.tqdm(data['family_accession'].unique()):
        indices = data[data['family_accession'] == cls].index.tolist()
        count = len(indices)

        # Logic for assigning indices to train/dev/test
        if count == 1:
            test_set.append(indices[0])
        elif count == 2:
            dev_set.append(indices[0])
            test_set.append(indices[1])
        elif count == 3:
            train_set.append(indices[0])
            dev_set.append(indices[1])
            test_set.append(indices[2])
        else:
            # Stratified split
            train_indices, test_indices = train_test_split(indices, test_size=0.2, stratify=[cls]*count)
            dev_indices, test_indices = train_test_split(test_indices, test_size=0.5, stratify=[cls]*len(test_indices))
            train_set.extend(train_indices)
            dev_set.extend(dev_indices)
            test_set.extend(test_indices)

    
    # Step 5: Convert index lists to numpy arrays
    train_indices = np.array(train_set)
    dev_indices = np.array(dev_set)
    test_indices = np.array(test_set)

    # Step 6: Create DataFrames from the selected indices
    train_df = data.iloc[train_indices].reset_index(drop=True)
    dev_df = data.iloc[dev_indices].reset_index(drop=True)
    test_df = data.iloc[test_indices].reset_index(drop=True)

    # Step 7: Drop unused columns: family_id, sequence_name, etc.
    train_df = train_df.drop(columns=['family_id', 'sequence_name'])
    dev_df = dev_df.drop(columns=['family_id', 'sequence_name'])
    test_df = test_df.drop(columns=['family_id', 'sequence_name'])

    # Step 8: Save train/dev/test datasets as CSV
    train_df.to_csv(f"{output_dir}/train.csv", index=False)
    dev_df.to_csv(f"{output_dir}/dev.csv", index=False)
    test_df.to_csv(f"{output_dir}/test.csv", index=False)

    # Step 9: Calculate class weights from the training set
    class_counts = train_df['family_accession'].value_counts()
    class_weights = {cls: len(train_df) / (len(class_counts) * count) for cls, count in class_counts.items()}

    # Step 10: Normalize weights and scale
    total_weight = sum(class_weights.values())
    class_weights = {cls: weight / total_weight for cls, weight in class_weights.items()}
    

    # Step 11: Save the class weights
    with open(f"{output_dir}/class_weights.txt", "w") as f:
        for cls, weight in class_weights.items():
            f.write(f"{cls}: {weight}\n")

    pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess protein data")
    parser.add_argument("--data_file", type=str, required=True, help="Path to train CSV file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the preprocessed files")
    args = parser.parse_args()

    preprocess_data(args.data_file, args.output_dir)
