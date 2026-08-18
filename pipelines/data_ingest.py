import os
import shutil
import zipfile
from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd


class DataIngestor:
    """Handles dataset discovery, multi-format file loading, ZIP archive extraction

    for tabular datasets, and extraction of image assets into ingested storage.
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

    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
    IGNORE_FILENAMES = {"readme", "license", "notes", "description"}

    def __init__(
        self,
        datasets_dir: Union[str, Path] = "datasets/raw_datasets",
        output_dir: Union[str, Path] = "datasets/ingested_dataset",
        extract_dir: Union[str, Path] = "datasets/extracted_temp",
    ):
        self.datasets_path = Path(datasets_dir)
        self.output_path = Path(output_dir)
        self.extract_path = Path(extract_dir)
        self.images_output_path = self.output_path / "images"

    def _is_readme_or_metadata(self, file_path: Union[str, Path]) -> bool:
        """Checks if a file is a readme, documentation, or metadata text file."""
        stem = Path(file_path).stem.lower()
        return any(keyword in stem for keyword in self.IGNORE_FILENAMES)

    def run(self) -> Dict[str, pd.DataFrame]:
        """Main entry point to scan, unpack, load, and save all datasets."""
        if not self.datasets_path.exists():
            raise FileNotFoundError(
                f"Directory '{self.datasets_path.resolve()}' does not exist."
            )

        self.output_path.mkdir(parents=True, exist_ok=True)
        all_dataframes: Dict[str, pd.DataFrame] = {}

        for file_path in sorted(self.datasets_path.iterdir()):
            if file_path.is_file():
                if self._is_readme_or_metadata(file_path):
                    print(f"[SKIP] Non-tabular metadata/readme file: {file_path.name}")
                    continue

                print(f"\n[PROCESSING] {file_path.name}...")
                loaded_dfs = self._ingest_file(file_path)
                all_dataframes.update(loaded_dfs)

        self._save_dataframes(all_dataframes)
        return all_dataframes

    def _ingest_file(self, file_path: Path) -> Dict[str, pd.DataFrame]:
        """Detects file format and routes to archive handler or single file loader."""
        if not file_path.exists():
            print(f"[ERROR] File not found: {file_path}")
            return {}

        if zipfile.is_zipfile(file_path):
            return self._handle_zip(file_path)

        if file_path.suffix.lower() in self.IMAGE_EXTENSIONS:
            self._handle_single_image(file_path)
            return {}

        df = self._load_single_file(file_path)
        return {file_path.stem: df} if df is not None else {}

    def _handle_zip(self, zip_path: Path) -> Dict[str, pd.DataFrame]:
        """Extracts ZIP contents using extended paths on Windows to bypass MAX_PATH limits,

        persisting image files and loading tabular data.
        """
        print(f"[INFO] '{zip_path.name}' is a zip file. Extracting...")
        target_dir = self.extract_path / zip_path.stem
        target_dir.mkdir(parents=True, exist_ok=True)

        # Bypass MAX_PATH on Windows by prepending \\?\
        extract_target_str = str(target_dir.resolve())
        if os.name == "nt" and not extract_target_str.startswith("\\\\?\\"):
            extract_target_str = "\\\\?\\" + extract_target_str

        # Prepare the image destination root with \\?\ support
        self.images_output_path.mkdir(parents=True, exist_ok=True)
        images_dest_str = str(self.images_output_path.resolve())
        if os.name == "nt" and not images_dest_str.startswith("\\\\?\\"):
            images_dest_str = "\\\\?\\" + images_dest_str

        dataframes: Dict[str, pd.DataFrame] = {}
        image_count = 0

        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_target_str)

            # Walk the extended extraction target path
            for root, _, files in os.walk(extract_target_str):
                for fname in files:
                    full_path = Path(root) / fname
                    ext = full_path.suffix.lower()

                    # Skip README or documentation text files inside the archive
                    if self._is_readme_or_metadata(full_path):
                        print(f"[SKIP] Non-tabular metadata/readme inside archive: {fname}")
                        continue

                    # Handle image files within archives
                    if ext in self.IMAGE_EXTENSIONS:
                        dest_file_str = os.path.join(images_dest_str, fname)
                        shutil.copy2(str(full_path), dest_file_str)
                        image_count += 1

                    # Handle tabular files within archives
                    elif ext in self.SUPPORTED_LOADERS:
                        try:
                            df = self._load_single_file(full_path)
                            if df is not None:
                                dataframes[full_path.stem] = df
                        except Exception as e:
                            print(f"[WARN] Skipped tabular '{fname}': {e}")

            if image_count > 0:
                print(f"[INFO] Extracted {image_count} images to '{self.images_output_path}'")

        finally:
            shutil.rmtree(extract_target_str, ignore_errors=True)

        return dataframes

    def _handle_single_image(self, file_path: Path) -> None:
        """Copies individual image files to the ingested images folder."""
        self.images_output_path.mkdir(parents=True, exist_ok=True)
        dest_file = self.images_output_path / file_path.name
        shutil.copy2(file_path, dest_file)
        print(f"[INFO] Saved image: {dest_file.name}")

    def _load_single_file(self, file_path: Path) -> Optional[pd.DataFrame]:
        """Loads supported tabular files into a pandas DataFrame."""
        if self._is_readme_or_metadata(file_path):
            print(f"[SKIP] Non-tabular metadata/readme file: {file_path.name}")
            return None

        ext = file_path.suffix.lower()
        if ext not in self.SUPPORTED_LOADERS:
            print(f"[SKIP] Unsupported file format: {file_path.name}")
            return None

        print(f"[INFO] Loading '{file_path.name}' as {ext} file...")
        return self.SUPPORTED_LOADERS[ext](file_path)

    def _save_dataframes(self, dataframes: Dict[str, pd.DataFrame]) -> None:
        """Saves processed tabular DataFrames as standard CSVs in output directory."""
        if not dataframes:
            print("\n[INFO] No tabular datasets to save.")
            return

        print(f"\n[INFO] Saving ingested datasets to '{self.output_path.resolve()}'...")
        for key, df in dataframes.items():
            clean_name = f"{key}_ingested.csv"
            save_file = self.output_path / clean_name
            df.to_csv(save_file, index=False)
            print(f"  Saved: {save_file.name} | Shape: {df.shape}")


class DatasetCombiner:
    """Consolidates PdM tabular datasets across telemetry, metadata, and event tables."""

    def __init__(self, data_sources: Union[str, Path, Dict[str, pd.DataFrame]]):
        if isinstance(data_sources, (str, Path)):
            self.dfs = self._load_from_directory(Path(data_sources))
        elif isinstance(data_sources, dict):
            self.dfs = data_sources
        else:
            raise ValueError("data_sources must be a folder path or DataFrame dictionary.")

    def _load_from_directory(self, folder_path: Path) -> Dict[str, pd.DataFrame]:
        """Loads ingested CSV files and strips prefix/suffix naming conventions."""
        dfs = {}
        for file_path in folder_path.glob("*.csv"):
            if "combined" in file_path.stem.lower():
                continue
            name = self._normalize_name(file_path.stem)
            dfs[name] = pd.read_csv(file_path)
        return dfs

    @staticmethod
    def _normalize_name(key: str) -> str:
        return (
            key.lower()
            .replace("_ingested", "")
            .replace("pdm_", "")
            .replace("microsoft_azure_pmd_raw_v1_", "")
            .strip()
        )

    def _prepare_timestamps(self) -> None:
        """Parses datetime strings into datetime objects."""
        time_series_keys = ["telemetry", "failures", "errors", "maint"]
        for key in time_series_keys:
            if key in self.dfs and "datetime" in self.dfs[key].columns:
                self.dfs[key]["datetime"] = pd.to_datetime(self.dfs[key]["datetime"])

    def combine(self) -> Optional[pd.DataFrame]:
        """Merges telemetry, machine properties, failures, errors, and maintenance records."""
        normalized_dfs = {self._normalize_name(k): v for k, v in self.dfs.items()}

        required_tables = ["telemetry", "machines"]
        for tbl in required_tables:
            if tbl not in normalized_dfs:
                print(f"[WARN] Missing '{tbl}' table. Skipping merge step.")
                return None

        self.dfs = normalized_dfs
        self._prepare_timestamps()

        # Merge base telemetry and machine metadata
        df_combined = pd.merge(
            self.dfs["telemetry"],
            self.dfs["machines"],
            on="machineID",
            how="left",
        )

        # Merge event tables (failures, errors, maint)
        for table_name in ["failures", "errors", "maint"]:
            if table_name in self.dfs:
                event_df = self.dfs[table_name]
                event_df = event_df.drop_duplicates(subset=["datetime", "machineID"])

                df_combined = pd.merge(
                    df_combined,
                    event_df,
                    on=["datetime", "machineID"],
                    how="left",
                )

        print(f"[SUCCESS] Combined dataset generated with shape: {df_combined.shape}")
        return df_combined


if __name__ == "__main__":
    # 1. Ingest all raw archives
    ingestor = DataIngestor(
        datasets_dir="datasets/raw_datasets",
        output_dir="datasets/ingested_dataset",
    )
    ingested_data = ingestor.run()

    # 2. Combine tabular data if available
    combiner = DatasetCombiner(data_sources=ingested_data)
    combined_df = combiner.combine()

    # 3. Save final merged CSV
    if combined_df is not None:
        output_merged_path = Path("datasets/ingested_dataset/PdM_combined.csv")
        combined_df.to_csv(output_merged_path, index=False)
        print(f"Saved merged dataset to '{output_merged_path.resolve()}'")