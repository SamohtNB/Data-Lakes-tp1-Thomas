import pandas as pd
import boto3
import os
from botocore.config import Config
from transformers import AutoTokenizer
import ast

def process_to_curated(bucket_staging, bucket_curated, input_file, output_file):
    # Init client S3 (LocalStack)
    s3 = boto3.client(
        "s3",
        endpoint_url="http://localhost:4566",
        aws_access_key_id="root",
        aws_secret_access_key="root",
        region_name="us-east-1",
        config=Config(signature_version="s3v4"),
        verify=False
    )

    # Chemin local temporaire
    os.makedirs("tmp", exist_ok=True)
    local_input_path = f"tmp/{input_file}"
    local_output_path = f"tmp/{output_file}"

    # Télécharger le fichier staging
    s3.download_file(bucket_staging, input_file, local_input_path)

    # Charger les données
    print("📥 Loading data")
    df = pd.read_csv(local_input_path)

    # Charger le tokenizer ESM
    print("🔡 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")

    # Tokeniser la colonne sequence
    print("🧬 Tokenizing sequences")
    df['input_ids'] = df['sequence'].apply(lambda x: tokenizer(x, truncation=True, padding='max_length', return_tensors='pt')['input_ids'][0].tolist())

    # Sauvegarde locale
    df.to_csv(local_output_path, index=False)

    # Upload vers le bucket curated
    print(f"📤 Uploading to s3://{bucket_curated}/{output_file}")
    s3.upload_file(local_output_path, bucket_curated, output_file)

    print("✅ Done: Curated file uploaded.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process staging data to curated zone")
    parser.add_argument("--bucket_staging", type=str, required=True, help="S3 staging bucket name")
    parser.add_argument("--bucket_curated", type=str, required=True, help="S3 curated bucket name")
    parser.add_argument("--input_file", type=str, required=True, help="Input CSV file in staging")
    parser.add_argument("--output_file", type=str, required=True, help="Output CSV file name for curated")

    args = parser.parse_args()

    process_to_curated(args.bucket_staging, args.bucket_curated, args.input_file, args.output_file)
