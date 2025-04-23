import os
import pandas as pd
import boto3

def unpack_to_raw(input_dir, bucket_name, output_file_name):
    """
    Unpacks and combines multiple CSV files from a directory into a single CSV file.

    Parameters:
    input_dir (str): Path to the directory containing the CSV files.
    output_file (str): Path to the output combined CSV file.
    """

    # Step 1: Initialize an empty list to store DataFrames
    list_data = []

    for root, dirs, files in os.walk(input_dir):
        print(f"Processing directory: {root}")
        for file in files:
            if file.endswith(".csv") or file.startswith("data-"):
                
                # Step 4: Read the CSV file using pandas
                file_path = os.path.join(root, file)
                data_temp = pd.read_csv(file_path, index_col=0)
                
                
                # Step 5: Append the DataFrame to the list
                list_data.append(data_temp)
    # Step 6: Concatenate all DataFrames
    if list_data:
        # Check if the list is not empty before concatenating
        print("Concatenating DataFrames...")
        data = pd.concat(list_data, axis=0, join="outer")
        
        tmp_dir = os.path.join("tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        
        combined_csv_path = os.path.join(tmp_dir, output_file_name)
        data.to_csv(combined_csv_path, index=True)
        print(f"Combined CSV file temporarly created at: {combined_csv_path}")
        
        s3 = boto3.client(
            "s3",
            endpoint_url="http://localhost:4566",  # ✅ PAS de HTTPS ici !
            aws_access_key_id="root",
            aws_secret_access_key="root",
            region_name="us-east-1",
            verify=False  # ✅ Ignore la vérification SSL
        )
        
        s3.upload_file(
            Filename=combined_csv_path,
            Bucket=bucket_name,
            Key=output_file_name
        )
                
        print(f"Combined CSV file uploaded to S3 bucket '{bucket_name}' with name '{output_file_name}'")
    else:
        # If the list is empty, create an empty DataFrame
        print("No CSV files found in the directory.")
        return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Unpack and combine protein data")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to input directory")
    parser.add_argument("--bucket_name", type=str, required=True, help="Name of the S3 bucket")
    parser.add_argument("--output_file_name", type=str, required=True, help="Path to output combined CSV file")
    args = parser.parse_args()

    unpack_to_raw(args.input_dir, args.bucket_name, args.output_file_name)
