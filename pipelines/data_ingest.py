import os
import shutil
import zipfile
from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd


class DataIngestor:
    """Handles dataset discovery, multi-format file loading, ZIP archive extraction,
    and output caching to CSV.
    """

    SUPPORTED_LOADERS = {
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

    def __init__(
        self,
        datasets_dir: Union[str, Path] = "datasets/raw_datasets",
        output_dir: Union[str, Path] = "datasets/ingested_dataset",
        extract_dir: Union[str, Path] = "datasets/extracted_temp",
    ):
        self.datasets_path = Path(datasets_dir)
        self.output_path = Path(output_dir)
        self.extract_path = Path(extract_dir)

    def run(self) -> Dict[str, pd.DataFrame]:
        """Main entry point to scan, load, and save all datasets."""
        if not self.datasets_path.exists():
            raise FileNotFoundError(
                f"Directory '{self.datasets_path.resolve()}' does not exist. "
                f"Please verify the folder path exists."
            )

        self.output_path.mkdir(parents=True, exist_ok=True)
        all_dataframes: Dict[str, pd.DataFrame] = {}

        for file_path in self.datasets_path.iterdir():
            if file_path.is_file():
                print(f"\n[PROCESSING] {file_path.name}...")
                loaded_dfs = self._ingest_file(file_path)
                all_dataframes.update(loaded_dfs)

        self._save_dataframes(all_dataframes)
        return all_dataframes

    def _ingest_file(self, file_path: Path) -> Dict[str, pd.DataFrame]:
        """Detects file format and delegates loading or extraction."""
        if not file_path.exists():
            print(f"[ERROR] File not found: {file_path}")
            return {}

        if zipfile.is_zipfile(file_path):
            return self._handle_zip(file_path)

        df = self._load_single_file(file_path)
        return {file_path.stem: df} if df is not None else {}

    def _handle_zip(self, zip_path: Path) -> Dict[str, pd.DataFrame]:
        """Extracts ZIP archive contents and processes files recursively."""
        print(f"[INFO] '{zip_path.name}' is a zip file. Extracting...")
        target_dir = self.extract_path / zip_path.stem
        target_dir.mkdir(parents=True, exist_ok=True)

        dataframes = {}
        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(target_dir)

            for root, _, files in os.walk(target_dir):
                for fname in files:
                    full_path = Path(root) / fname
                    try:
                        df = self._load_single_file(full_path)
                        if df is not None:
                            dataframes[full_path.stem] = df
                    except Exception as e:
                        print(f"[WARN] Skipped '{fname}': {e}")
        finally:
            shutil.rmtree(target_dir, ignore_errors=True)

        return dataframes

    def _load_single_file(self, file_path: Path) -> Optional[pd.DataFrame]:
        """Loads a dataset into a DataFrame based on file extension."""
        ext = file_path.suffix.lower()
        if ext not in self.SUPPORTED_LOADERS:
            print(f"[SKIP] Unsupported file format: {file_path.name}")
            return None

        print(f"[INFO] Loading '{file_path.name}' as {ext} file...")
        return self.SUPPORTED_LOADERS[ext](file_path)

    def _save_dataframes(self, dataframes: Dict[str, pd.DataFrame]) -> None:
        """Saves processed DataFrames into standard ingested CSV files."""
        print(
            f"\n[INFO] Saving ingested datasets to '{self.output_path.resolve()}'..."
        )
        for key, df in dataframes.items():
            clean_name = f"{key}_ingested.csv"
            save_file = self.output_path / clean_name
            df.to_csv(save_file, index=False)
            print(f" Saved: {save_file.name} | Shape: {df.shape}")


class DatasetCombiner:
    """Encapsulates loading, timestamp parsing, and left-joining Predictive Maintenance datasets."""

    def __init__(self, data_sources: Union[str, Path, Dict[str, pd.DataFrame]]):
        if isinstance(data_sources, (str, Path)):
            self.dfs = self._load_from_directory(Path(data_sources))
        elif isinstance(data_sources, dict):
            self.dfs = data_sources
        else:
            raise ValueError(
                "data_sources must be a folder path or dictionary of DataFrames."
            )

    def _load_from_directory(self, folder_path: Path) -> Dict[str, pd.DataFrame]:
        """Utility to load matching CSV files from disk if raw paths are supplied."""
        required_keys = ["telemetry", "machines", "failures", "errors", "maint"]
        dfs = {}

        for file_path in folder_path.glob("*.csv"):
            name = file_path.stem.lower().replace("_ingested", "").replace("pdm_", "")
            if name in required_keys:
                dfs[name] = pd.read_csv(file_path)

        return dfs

    def _prepare_timestamps(self) -> None:
        """Converts datetime columns across time-series DataFrames to datetime objects."""
        time_series_keys = ["telemetry", "failures", "errors", "maint"]
        for key in time_series_keys:
            if key in self.dfs and "datetime" in self.dfs[key].columns:
                self.dfs[key]["datetime"] = pd.to_datetime(
                    self.dfs[key]["datetime"]
                )

    def combine(self) -> pd.DataFrame:
        """Merges telemetry, machine metadata, failures, errors, and maintenance records."""
        normalized_dfs = {
            k.lower().replace("pdm_", "").replace("_ingested", ""): v
            for k, v in self.dfs.items()
        }

        required_tables = ["telemetry", "machines"]
        for tbl in required_tables:
            if tbl not in normalized_dfs:
                raise KeyError(
                    f"Missing required table '{tbl}' for dataset combination."
                )

        self.dfs = normalized_dfs
        self._prepare_timestamps()

        # Step 1: Base merge between telemetry and static machine properties
        df_combined = pd.merge(
            self.dfs["telemetry"],
            self.dfs["machines"],
            on="machineID",
            how="left",
        )

        # Step 2: Merge event tables using composite key [datetime, machineID]
        event_tables = ["failures", "errors", "maint"]
        for table_name in event_tables:
            if table_name in self.dfs:
                df_combined = pd.merge(
                    df_combined,
                    self.dfs[table_name],
                    on=["datetime", "machineID"],
                    how="left",
                )

        print(
            f"[SUCCESS] Combined dataset generated with shape: {df_combined.shape}"
        )
        return df_combined


# --- Main Execution Pipeline ---
if __name__ == "__main__":
    # 1. Ingest raw datasets from 'datasets/raw_datasets' and output to 'datasets/ingested_dataset'
    ingestor = DataIngestor(
        datasets_dir="datasets/raw_datasets",
        output_dir="datasets/ingested_dataset",
    )
    ingested_data = ingestor.run()

    # 2. Combine the ingested DataFrames into a single merged dataset
    combiner = DatasetCombiner(data_sources=ingested_data)
    combined_df = combiner.combine()

    # 3. Save final merged dataset into 'datasets/ingested_dataset'
    output_merged_path = Path("datasets/ingested_dataset/PdM_combined.csv")
    combined_df.to_csv(output_merged_path, index=False)
    print(f"Saved merged dataset to '{output_merged_path.resolve()}'")