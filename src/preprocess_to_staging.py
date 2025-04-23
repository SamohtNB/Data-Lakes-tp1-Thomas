import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import tqdm
import joblib
from collections import OrderedDict
import boto3
import os
from botocore.config import Config


def preprocess_data(bucket_raw, bucket_staging, input_file, output_prefix):
    # Configuration du client S3 pour LocalStack (avec HTTPS désactivé)
    s3 = boto3.client(
        "s3",
        endpoint_url="http://localhost:4566",
        aws_access_key_id="root",
        aws_secret_access_key="root",
        region_name="us-east-1",
        config=Config(signature_version="s3v4"),
        verify=False
    )

    # Créer un dossier temporaire pour les fichiers intermédiaires
    os.makedirs("tmp", exist_ok=True)

    # Télécharger le fichier raw depuis S3 vers un chemin local
    local_raw_path = f"tmp/{input_file}"
    s3.download_file(bucket_raw, input_file, local_raw_path)

    # Lire le fichier CSV téléchargé
    print('📥 Loading Data')
    data = pd.read_csv(local_raw_path)

    # Supprimer les lignes avec valeurs manquantes
    data = data.dropna()

    # Encoder la colonne "family_accession" en classes numériques
    label_encoder = LabelEncoder()
    data['class_encoded'] = label_encoder.fit_transform(data['family_accession'])

    # Sauvegarde locale du label encoder (utile pour l'entraînement ultérieur)
    joblib.dump(label_encoder, f'tmp/{output_prefix}_label_encoder.joblib')

    # Extraire les valeurs pour traitement
    family_accession = data['family_accession'].values
    class_encoded = data['class_encoded'].values

    # Trouver toutes les classes uniques
    unique_classes, _ = np.unique(family_accession, return_counts=True)

    # Listes pour stocker les indices des splits
    train_indices, dev_indices, test_indices = [], [], []

    # Répartition des données par classe
    print("🔀 Distributing data by class")
    for cls in tqdm.tqdm(unique_classes):
        class_data_indices = np.where(family_accession == cls)[0]
        count = len(class_data_indices)

        # Logique de répartition en fonction du nombre d'exemples par classe
        if count == 1:
            test_indices.extend(class_data_indices)
        elif count == 2:
            dev_indices.extend(class_data_indices[:1])
            test_indices.extend(class_data_indices[1:])
        elif count == 3:
            train_indices.extend(class_data_indices[:1])
            dev_indices.extend(class_data_indices[1:2])
            test_indices.extend(class_data_indices[2:])
        else:
            # Split stratifié pour les classes > 3 éléments
            train_part, remaining = train_test_split(
                class_data_indices,
                test_size=2/3,
                random_state=42,
                stratify=class_encoded[class_data_indices]
            )
            dev_part, test_part = train_test_split(
                remaining,
                test_size=0.5,
                random_state=42,
                stratify=class_encoded[remaining]
            )
            train_indices.extend(train_part)
            dev_indices.extend(dev_part)
            test_indices.extend(test_part)

    # Construire les DataFrames pour chaque split
    train_df = data.iloc[train_indices].drop(columns=["family_id", "sequence_name", "family_accession"])
    dev_df = data.iloc[dev_indices].drop(columns=["family_id", "sequence_name", "family_accession"])
    test_df = data.iloc[test_indices].drop(columns=["family_id", "sequence_name", "family_accession"])

    # Sauvegarder localement les fichiers
    train_path = f"tmp/{output_prefix}_train.csv"
    dev_path = f"tmp/{output_prefix}_dev.csv"
    test_path = f"tmp/{output_prefix}_test.csv"

    train_df.to_csv(train_path, index=False)
    dev_df.to_csv(dev_path, index=False)
    test_df.to_csv(test_path, index=False)

    # Uploader les fichiers vers le bucket "staging"
    print(f"📤 Uploading to s3://{bucket_staging}/")
    s3.upload_file(train_path, bucket_staging, f"{output_prefix}_train.csv")
    s3.upload_file(dev_path, bucket_staging, f"{output_prefix}_dev.csv")
    s3.upload_file(test_path, bucket_staging, f"{output_prefix}_test.csv")

    print(f"✅ Done: preprocessed files available in '{bucket_staging}'")


# Entrée point principal pour exécution CLI
if __name__ == "__main__":
    import argparse

    # Définition des arguments en ligne de commande
    parser = argparse.ArgumentParser(description="Preprocess protein data to staging zone")
    parser.add_argument("--bucket_raw", type=str, required=True, help="S3 raw bucket name")
    parser.add_argument("--bucket_staging", type=str, required=True, help="S3 staging bucket name")
    parser.add_argument("--input_file", type=str, required=True, help="Name of the input file in raw bucket")
    parser.add_argument("--output_prefix", type=str, required=True, help="Prefix for the output files")
    args = parser.parse_args()

    # Lancement du prétraitement
    preprocess_data(args.bucket_raw, args.bucket_staging, args.input_file, args.output_prefix)
