import os
import zipfile
import shutil
import pandas as pd
from pathlib import Path

# TODO : go to 'datasets' folder and checks every file and starts ingesting 

def ingest_data(file_path, extract_dir="extracted_data"):
    """
    Detects file format and loads data accordingly.
    If a zip file is provided, extracts it first and processes
    all supported files inside.

    Parameters:
        file_path (str): Path to the input file (csv, xlsx, json, parquet, txt, zip, etc.)
        extract_dir (str): Directory to extract zip contents into.

    Returns:
        dict: {filename: DataFrame} for all successfully loaded files.
    """
    file_path = Path(file_path)
    dataframes = {}

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # --- Handle ZIP files ---
    if zipfile.is_zipfile(file_path):
        print(f"[INFO] '{file_path.name}' is a zip file. Extracting...")
        extract_path = Path(extract_dir)
        extract_path.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

        print(f"[INFO] Extracted to: {extract_path.resolve()}")

        # Recursively process extracted files
        for root, _, files in os.walk(extract_path):
            for fname in files:
                full_path = Path(root) / fname
                try:
                    df = _load_single_file(full_path)
                    if df is not None:
                        dataframes[fname] = df
                except Exception as e:
                    print(f"[WARN] Skipped '{fname}': {e}")

        return dataframes

    # --- Handle single non-zip file ---
    df = _load_single_file(file_path)
    if df is not None:
        dataframes[file_path.name] = df
    return dataframes


def _load_single_file(file_path):
    """Loads a single file based on its extension."""
    ext = file_path.suffix.lower()

    loaders = {
        ".csv": lambda p: pd.read_csv(p),
        ".tsv": lambda p: pd.read_csv(p, sep="\t"),
        ".txt": lambda p: pd.read_csv(p, sep=None, engine="python"),  # auto-detect delimiter
        ".xlsx": lambda p: pd.read_excel(p),
        ".xls": lambda p: pd.read_excel(p),
        ".json": lambda p: pd.read_json(p),
        ".parquet": lambda p: pd.read_parquet(p),
        ".feather": lambda p: pd.read_feather(p),
        ".pkl": lambda p: pd.read_pickle(p),
    }

    if ext not in loaders:
        print(f"[SKIP] Unsupported file format: {file_path.name}")
        return None

    print(f"[INFO] Loading '{file_path.name}' as {ext} file...")
    return loaders[ext](file_path)


# --- Example usage ---
if __name__ == "__main__":
    data = ingest_data("sensor_data.zip")  # or "sensor_data.csv", etc.

    for name, df in data.items():
        print(f"\n=== {name} ===")
        print(df.head())
        print(f"Shape: {df.shape}")