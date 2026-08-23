import os
import shutil
import zipfile
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageStat
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from skimage.feature import graycomatrix, graycoprops

# Configure Device and Logging
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# ==========================================
# 1. IMAGE INGESTION MODULE
# ==========================================

class ImageIngestor:
    """Discovers raw images and extracts images from ZIP archives into ingested storage."""
    
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
        stem = Path(file_path).stem.lower()
        return any(keyword in stem for keyword in self.IGNORE_FILENAMES)

    def _handle_single_image(self, file_path: Path) -> None:
        self.images_output_path.mkdir(parents=True, exist_ok=True)
        dest_file = self.images_output_path / file_path.name
        shutil.copy2(file_path, dest_file)
        print(f"[INFO] Saved image: {dest_file.name}")

    def _handle_zip(self, zip_path: Path) -> None:
        print(f"[INFO] '{zip_path.name}' is a zip file. Extracting images...")
        target_dir = self.extract_path / zip_path.stem
        target_dir.mkdir(parents=True, exist_ok=True)

        # Bypass MAX_PATH on Windows by prepending \\?\
        extract_target_str = str(target_dir.resolve())
        if os.name == "nt" and not extract_target_str.startswith("\\\\?\\"):
            extract_target_str = "\\\\?\\" + extract_target_str

        self.images_output_path.mkdir(parents=True, exist_ok=True)
        images_dest_str = str(self.images_output_path.resolve())
        if os.name == "nt" and not images_dest_str.startswith("\\\\?\\"):
            images_dest_str = "\\\\?\\" + images_dest_str

        image_count = 0
        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_target_str)

            for root, _, files in os.walk(extract_target_str):
                for fname in files:
                    full_path = Path(root) / fname
                    ext = full_path.suffix.lower()

                    if self._is_readme_or_metadata(full_path):
                        continue

                    if ext in self.IMAGE_EXTENSIONS:
                        dest_file_str = os.path.join(images_dest_str, fname)
                        shutil.copy2(str(full_path), dest_file_str)
                        image_count += 1

            if image_count > 0:
                print(f"[INFO] Extracted {image_count} images to '{self.images_output_path}'")
        finally:
            shutil.rmtree(extract_target_str, ignore_errors=True)

    def run(self) -> None:
        if not self.datasets_path.exists():
            raise FileNotFoundError(f"Directory '{self.datasets_path.resolve()}' does not exist.")

        self.images_output_path.mkdir(parents=True, exist_ok=True)

        for file_path in sorted(self.datasets_path.iterdir()):
            if file_path.is_file():
                if self._is_readme_or_metadata(file_path):
                    continue
                if zipfile.is_zipfile(file_path):
                    self._handle_zip(file_path)
                elif file_path.suffix.lower() in self.IMAGE_EXTENSIONS:
                    self._handle_single_image(file_path)


# ==========================================
# 2. IMAGE PREPROCESSING & CLEANING
# ==========================================

def clean_ingested_images(
    images_dir: Union[str, Path] = "datasets/ingested_dataset/images",
    output_dir: Union[str, Path] = "datasets/clean_dataset/images",
    target_size: Tuple[int, int] = (256, 256),
    min_std_threshold: float = 1.0,
    valid_extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"),
) -> Dict[str, int]:
    """Removes corrupt/blank images, letterboxes to preserve aspect ratio, and saves as standard PNGs."""
    images_path = Path(images_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if not images_path.exists():
        logging.warning(f"Image directory '{images_path.resolve()}' does not exist.")
        return {"total": 0, "kept": 0, "removed_corrupt": 0, "removed_blank": 0}

    stats = {"total": 0, "kept": 0, "removed_corrupt": 0, "removed_blank": 0}
    image_files = sorted(f for f in images_path.iterdir() if f.is_file() and f.suffix.lower() in valid_extensions)

    resample_filter = getattr(Image.Resampling, "LANCZOS", Image.LANCZOS)

    for file_path in image_files:
        stats["total"] += 1
        try:
            with Image.open(file_path) as img:
                img.verify()
            with Image.open(file_path) as img:
                img_rgb = img.convert("RGB")
                
                # Check variance across channels to detect blank/flat images
                stat = ImageStat.Stat(img_rgb)
                avg_std = sum(stat.stddev) / len(stat.stddev)
                if avg_std < min_std_threshold:
                    stats["removed_blank"] += 1
                    logging.info(f"[SKIP] Low-variance image: {file_path.name} (std={avg_std:.2f})")
                    continue

                # Resize with aspect-ratio preservation (Letterbox/Pad)
                img_resized = ImageOps.pad(img_rgb, target_size, method=resample_filter, color=(0, 0, 0))
                dest_file = out_path / f"{file_path.stem}.png"
                img_resized.save(dest_file, format="PNG")
                stats["kept"] += 1

        except (OSError, IOError, ValueError) as e:
            stats["removed_corrupt"] += 1
            logging.warning(f"[REMOVE] Corrupt/unreadable image '{file_path.name}': {e}")
            continue

    logging.info(
        f"Image cleaning complete: {stats['kept']}/{stats['total']} kept | "
        f"corrupt={stats['removed_corrupt']}, blank={stats['removed_blank']}"
    )
    return stats


# ==========================================
# 3. FEATURE ENGINEERING HELPERS
# ==========================================

# 3.1. Padding Exclusion Mask
def get_padding_mask(img_rgb: np.ndarray, black_thresh: int = 8) -> np.ndarray:
    """Builds a binary mask (1=real content, 0=artificial black letterbox pad)."""
    is_black = np.all(img_rgb <= black_thresh, axis=-1)
    return (~is_black).astype(np.uint8)

def masked_pixels(img_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Returns only unpadded pixels as an (N, 3) array."""
    return img_rgb[mask.astype(bool)]

# 3.2. Rust Ratio via HSV Thresholding
RUST_HSV_LOWER = np.array([3, 60, 40])
RUST_HSV_UPPER = np.array([25, 255, 220])
METAL_HSV_LOWER = np.array([0, 0, 50])
METAL_HSV_UPPER = np.array([179, 50, 220])

def compute_rust_ratio(img_rgb: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """Thresholds rust vs healthy-metal pixels within unpadded mask."""
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    rust_mask = cv2.inRange(hsv, RUST_HSV_LOWER, RUST_HSV_UPPER)
    metal_mask = cv2.inRange(hsv, METAL_HSV_LOWER, METAL_HSV_UPPER)
    
    valid = mask.astype(bool)
    rust_mask = (rust_mask.astype(bool) & valid)
    metal_mask = (metal_mask.astype(bool) & valid)
    
    valid_pixel_count = max(valid.sum(), 1)
    rust_pixels = int(rust_mask.sum())
    metal_pixels = int(metal_mask.sum())
    
    return {
        "rust_pixel_ratio": rust_pixels / valid_pixel_count,
        "rust_to_metal_ratio": rust_pixels / max(metal_pixels, 1),
        "metal_pixel_ratio": metal_pixels / valid_pixel_count,
    }

# 3.3. Mask-Aware Color/Brightness Statistics
def compute_color_stats(img_rgb: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """Computes brightness, variance, and RGB stats on unpadded pixels only."""
    pixels = masked_pixels(img_rgb, mask)
    if pixels.size == 0:
        return {
            "mean_brightness": 0.0, "std_brightness": 0.0,
            "mean_r": 0.0, "mean_g": 0.0, "mean_b": 0.0,
            "valid_pixel_fraction": 0.0,
        }
    brightness = pixels.mean(axis=1)
    total_pixels = mask.size
    return {
        "mean_brightness": float(brightness.mean()),
        "std_brightness": float(brightness.std()),
        "mean_r": float(pixels[:, 0].mean()),
        "mean_g": float(pixels[:, 1].mean()),
        "mean_b": float(pixels[:, 2].mean()),
        "valid_pixel_fraction": float(mask.sum() / total_pixels),
    }

# 3.4. Texture (GLCM) and Edge Descriptors
def compute_texture_features(
    gray: np.ndarray,
    mask: np.ndarray,
    distances=(1, 3),
    angles=(0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
) -> Dict[str, float]:
    """Computes GLCM texture metrics (contrast, correlation, energy, homogeneity, entropy)."""
    valid = mask.astype(bool)
    if valid.sum() < 100:
        return {
            "glcm_contrast": 0.0, "glcm_correlation": 0.0,
            "glcm_energy": 0.0, "glcm_homogeneity": 0.0,
            "glcm_entropy": 0.0,
        }
    gray_masked = np.where(valid, gray, 0).astype(np.uint8)
    levels = 32
    gray_quantized = (gray_masked.astype(np.float32) / 255 * (levels - 1)).astype(np.uint8)
    
    glcm = graycomatrix(
        gray_quantized, distances=list(distances), angles=list(angles),
        levels=levels, symmetric=True, normed=True
    )
    contrast = graycoprops(glcm, "contrast").mean()
    correlation = graycoprops(glcm, "correlation").mean()
    energy = graycoprops(glcm, "energy").mean()
    homogeneity = graycoprops(glcm, "homogeneity").mean()
    
    glcm_probs = glcm.astype(np.float64)
    glcm_probs = glcm_probs / (glcm_probs.sum() + 1e-10)
    entropy = -np.sum(glcm_probs * np.log2(glcm_probs + 1e-10))
    
    return {
        "glcm_contrast": float(contrast),
        "glcm_correlation": float(correlation),
        "glcm_energy": float(energy),
        "glcm_homogeneity": float(homogeneity),
        "glcm_entropy": float(entropy),
    }

def compute_edge_density(gray: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """Computes Sobel gradients and Canny edge density over the valid region."""
    valid = mask.astype(bool)
    valid_pixel_count = max(valid.sum(), 1)
    
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    sobel_mag_masked = sobel_mag * valid
    
    canny_edges = cv2.Canny(gray, threshold1=50, threshold2=150)
    canny_masked = (canny_edges > 0) & valid
    
    return {
        "sobel_mean_gradient": float(sobel_mag_masked.sum() / valid_pixel_count),
        "canny_edge_density": float(canny_masked.sum() / valid_pixel_count),
    }

# 3.5. CNN Feature Extractor
class CNNEmbedder:
    """Extracts a dense 2048-dim feature vector per image using a frozen pre-trained ResNet50."""
    def __init__(self, model_name: str = "resnet50"):
        if model_name == "resnet50":
            backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        elif model_name == "mobilenet_v3":
            backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
            self.feature_extractor = nn.Sequential(backbone.features, nn.AdaptiveAvgPool2d(1))
        else:
            raise ValueError(f"Unsupported model name: {model_name}")

        self.feature_extractor.eval().to(device)
        for p in self.feature_extractor.parameters():
            p.requires_grad = False

        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def embed(self, img_rgb: np.ndarray) -> np.ndarray:
        pil_img = Image.fromarray(img_rgb)
        tensor = self.transform(pil_img).unsqueeze(0).to(device)
        feat = self.feature_extractor(tensor)
        return feat.squeeze().cpu().numpy().flatten()


# ==========================================
# 4. DATASET FEATURE TABLE GENERATOR
# ==========================================

def extract_image_features(image_path: Path, embedder: Optional[CNNEmbedder] = None) -> Dict:
    """Extracts domain-specific and deep visual features from a single cleaned image."""
    with Image.open(image_path) as img:
        img_rgb = np.array(img.convert("RGB"))
    
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    mask = get_padding_mask(img_rgb)
    
    rust_feats = compute_rust_ratio(img_rgb, mask)
    color_feats = compute_color_stats(img_rgb, mask)
    texture_feats = compute_texture_features(gray, mask)
    edge_feats = compute_edge_density(gray, mask)
    
    embedding_feats = {}
    if embedder is not None:
        emb = embedder.embed(img_rgb)
        embedding_feats = {f"cnn_emb_{i}": float(v) for i, v in enumerate(emb)}
        
    return {
        "image_filename": image_path.name,
        **rust_feats,
        **color_feats,
        **texture_feats,
        **edge_feats,
        **embedding_feats,
    }

def build_image_feature_table(
    images_dir: Union[str, Path],
    output_path: Union[str, Path],
    use_cnn_embeddings: bool = True,
    cnn_model_name: str = "resnet50",
    valid_extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg"),
) -> pd.DataFrame:
    """Iterates through clean images, generates features, and exports the final model dataset CSV."""
    output_path = Path(output_path)
    embedder = CNNEmbedder(model_name=cnn_model_name) if use_cnn_embeddings else None
    
    image_files = sorted(
        f for f in Path(images_dir).iterdir()
        if f.is_file() and f.suffix.lower() in valid_extensions
    )
    print(f"[INFO] Extracting visual features from {len(image_files)} images...")
    
    records = []
    for i, img_path in enumerate(image_files, 1):
        try:
            records.append(extract_image_features(img_path, embedder=embedder))
        except Exception as e:
            print(f"[WARN] Skipped '{img_path.name}': {e}")
        if i % 50 == 0:
            print(f"Processed {i}/{len(image_files)}...")
            
    df_img_features = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_img_features.to_csv(output_path, index=False)
    print(f"[SUCCESS] Saved image feature dataset to: '{output_path}' | shape={df_img_features.shape}")
    return df_img_features


# ==========================================
# 5. EXECUTION PIPELINE
# ==========================================

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent if "__file__" in locals() else Path.cwd()
    RAW_DIR = BASE_DIR / "datasets" / "raw_datasets"
    INGESTED_IMAGES_DIR = BASE_DIR / "datasets" / "ingested_dataset" / "images"
    CLEAN_IMAGES_DIR = BASE_DIR / "datasets" / "clean_dataset" / "images"
    FINAL_DATASET_CSV = BASE_DIR / "datasets" / "model_train_dataset" / "image_features.csv"

    # Step 1: Ingest images from standalone image files and ZIP archives
    ingestor = ImageIngestor(datasets_dir=RAW_DIR, output_dir=BASE_DIR / "datasets" / "ingested_dataset")
    ingestor.run()

    # Step 2: Clean images (filter blanks/corrupt, letterbox/pad to 256x256)
    clean_ingested_images(
        images_dir=INGESTED_IMAGES_DIR,
        output_dir=CLEAN_IMAGES_DIR,
        target_size=(256, 256),
    )

    # Step 3: Feature Engineering to create the model-ready dataset
    build_image_feature_table(
        images_dir=CLEAN_IMAGES_DIR,
        output_path=FINAL_DATASET_CSV,
        use_cnn_embeddings=True,
        cnn_model_name="resnet50",
    )