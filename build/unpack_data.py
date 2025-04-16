import os
import pandas as pd

def unpack_data(input_dir, output_file):
    """
    Unpacks and combines multiple CSV files from a directory into a single CSV file.

    Parameters:
    input_dir (str): Path to the directory containing the CSV files.
    output_file (str): Path to the output combined CSV file.
    """

    # Step 1: Initialize an empty list to store DataFrames
    list_data = []
    
    # Step 2: Loop over files in the input directory
    for file in os.listdir(input_dir):
        print(file)
        
        # Step 3: Check if the file is a CSV or matches a naming pattern
        if file.endswith(".csv") or file.startswith("data-"):
            
            # Step 4: Read the CSV file using pandas
            file_path = os.path.join(input_dir, file)
            data_temp = pd.read_csv(file_path, index_col=0)
            
            
            # Step 5: Append the DataFrame to the list
            list_data.append(data_temp)
            
    # Step 6: Concatenate all DataFrames
    data = pd.concat(list_data, axis=0, join="outer")
    
    # Step 7: Save the combined DataFrame to output_file
    data.to_csv(output_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Unpack and combine protein data")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to input directory")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output combined CSV file")
    args = parser.parse_args()

    unpack_data(args.input_dir, args.output_file)
