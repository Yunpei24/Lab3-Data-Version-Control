"""
This script takes raw data and outputs modified data (e.g., multiply one of the features with 2).
"""
import json
import os
import pandas as pd

def preprocess(input_path: str, output_path: str, add_row: bool = False):
    # Read the raw data
    df = pd.read_json(input_path)

    # Example modification: multiply the first feature by 2
    df.iloc[:, 0] = df.iloc[:, 0] * 2

    if add_row:
        # Add a new row with modified values
        new_row = df.iloc[0].copy()
        new_row["sepalLength"] *= 2
        df = df.append(new_row, ignore_index=True)


    # Save the modified data
    df.to_json(output_path, orient='records', lines=True)


if __name__ == "__main__":
    input_path = os.path.join("data", "iris.json")
    output_path = os.path.join("data", "iris_modified.json")
    preprocess(input_path, output_path)