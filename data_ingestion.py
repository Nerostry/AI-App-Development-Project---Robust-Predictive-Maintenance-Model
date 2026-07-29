import os
import zipfile
import shutil
import pandas as pd
from pathlib import Path

def ingest_datasets(datasets_dir="datasets", output_dir="ingested_datasets"):
    """
    Scans the input datasets directory, loads all supported files (and unzips archives),
    and saves the loaded DataFrames as CSVs into the output directory.

    Parameters:
        datasets_dir (str/Path): Directory containing raw dataset files/zips.
        output_dir (str/Path): Directory where ingested datasets will be stored.

    Returns:
        dict: {filename: DataFrame} for all successfully processed datasets.
    """
    datasets_path = Path(datasets_dir)
    output_path = Path(output_dir)

    if not datasets_path.exists():
        raise FileNotFoundError(
            f"Directory '{datasets_path.resolve()}' does not exist. "
            f"Please create it and place your dataset files inside."
        )

    output_path.mkdir(parents=True, exist_ok=True)
    all_dataframes = {}

    # Iterate through all files directly inside the datasets folder
    for file_path in datasets_path.iterdir():
        if file_path.is_file():
            print(f"\n[PROCESSING] {file_path.name}...")
            loaded_dfs = ingest_data(file_path)
            all_dataframes.update(loaded_dfs)

    # Save all ingested DataFrames into the output folder
    print(f"\n[INFO] Saving ingested datasets to '{output_path.resolve()}'...")
    for filename, df in all_dataframes.items():
        # Clean extension for saving back as standard CSV
        clean_name = Path(filename).stem + "_ingested.csv"
        save_file = output_path / clean_name
        
        df.to_csv(save_file, index=False)
        print(f" Saved: {save_file.name} | Shape: {df.shape}")

    return all_dataframes


def ingest_data(file_path, extract_dir="extracted_data"):
    """
    Detects file format and loads data accordingly.
    If a zip file is provided, extracts it first and processes
    all supported files inside.
    """
    file_path = Path(file_path)
    dataframes = {}

    if not file_path.exists():
        print(f"[ERROR] File not found: {file_path}")
        return dataframes

    # --- Handle ZIP files ---
    if zipfile.is_zipfile(file_path):
        print(f"[INFO] '{file_path.name}' is a zip file. Extracting...")
        extract_path = Path(extract_dir) / file_path.stem
        extract_path.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

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

        # Cleanup extracted temporary folder after ingestion
        shutil.rmtree(extract_path, ignore_errors=True)
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
        ".txt": lambda p: pd.read_csv(p, sep=None, engine="python"),
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


# --- Execution ---
if __name__ == "__main__":
    # Scans 'datasets/', processes zip/csv/excel files, and outputs to 'ingested_datasets/'
    data = ingest_datasets(datasets_dir="datasets", output_dir="ingested_datasets")