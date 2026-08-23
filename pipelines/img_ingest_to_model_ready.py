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

# Prevent OpenBLAS / OpenMP threading conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Suppress console logging to output only on completion
logging.basicConfig(level=logging.ERROR)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 1. IMAGE PREPROCESSING & FEATURE EXTRACTION HELPERS

def get_padding_mask(img_rgb: np.ndarray, black_thresh: int = 80) -> np.ndarray:
    is_black = np.all(img_rgb <= black_thresh, axis=-1)
    return (~is_black).astype(np.uint8)


def masked_pixels(img_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return img_rgb[mask.astype(bool)]


RUST_HSV_LOWER = np.array([3, 60, 40])
RUST_HSV_UPPER = np.array([25, 255, 220])
METAL_HSV_LOWER = np.array([0, 0, 50])
METAL_HSV_UPPER = np.array([179, 50, 220])


def compute_rust_ratio(img_rgb: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    rust_mask = cv2.inRange(hsv, RUST_HSV_LOWER, RUST_HSV_UPPER)
    metal_mask = cv2.inRange(hsv, METAL_HSV_LOWER, METAL_HSV_UPPER)
    
    valid = mask.astype(bool)
    rust_mask = rust_mask.astype(bool) & valid
    metal_mask = metal_mask.astype(bool) & valid
    
    valid_pixel_count = max(int(valid.sum()), 1)
    rust_pixels = int(rust_mask.sum())
    metal_pixels = int(metal_mask.sum())
    
    return {
        "rust_pixel_ratio": rust_pixels / valid_pixel_count,
        "rust_to_metal_ratio": rust_pixels / max(metal_pixels, 1),
        "metal_pixel_ratio": metal_pixels / valid_pixel_count,
    }


def compute_color_stats(img_rgb: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    pixels = masked_pixels(img_rgb, mask)
    if pixels.size == 0:
        return {
            "mean_brightness": 0.0, "std_brightness": 0.0,
            "mean_r": 0.0, "mean_g": 0.0, "mean_b": 0.0,
            "valid_pixel_fraction": 0.0,
        }
    brightness = pixels.mean(axis=1)
    return {
        "mean_brightness": float(brightness.mean()),
        "std_brightness": float(brightness.std()),
        "mean_r": float(pixels[:, 0].mean()),
        "mean_g": float(pixels[:, 1].mean()),
        "mean_b": float(pixels[:, 2].mean()),
        "valid_pixel_fraction": float(mask.sum() / mask.size),
    }


def compute_texture_features(
    gray: np.ndarray,
    mask: np.ndarray,
    distances: Tuple[int, ...] = (1, 3),
    angles: Tuple[float, ...] = (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
) -> Dict[str, float]:
    valid = mask.astype(bool)
    if valid.sum() <= 100:
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
    glcm_probs = glcm.astype(np.float64) / (glcm.sum() + 1e-10)
    
    return {
        "glcm_contrast": float(graycoprops(glcm, "contrast").mean()),
        "glcm_correlation": float(graycoprops(glcm, "correlation").mean()),
        "glcm_energy": float(graycoprops(glcm, "energy").mean()),
        "glcm_homogeneity": float(graycoprops(glcm, "homogeneity").mean()),
        "glcm_entropy": float(-np.sum(glcm_probs * np.log2(glcm_probs + 1e-10))),
    }


def compute_edge_density(gray: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    valid = mask.astype(bool)
    valid_pixel_count = max(int(valid.sum()), 1)
    
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2) * valid
    
    canny_edges = (cv2.Canny(gray, threshold1=50, threshold2=150) > 0) & valid
    
    return {
        "sobel_mean_gradient": float(sobel_mag.sum() / valid_pixel_count),
        "canny_edge_density": float(canny_edges.sum() / valid_pixel_count),
    }


# 2. CNN EMBEDDER

class CNNEmbedder:
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


# 3. PIPELINE PROCESSOR (RAW -> MODEL DATASET)

def process_raw_dataset_to_train(
    raw_dir: Union[str, Path],
    output_path: Union[str, Path],
    target_size: Tuple[int, int] = (256, 256),
    min_std_threshold: float = 1.0,
    use_cnn_embeddings: bool = True,
    cnn_model_name: str = "resnet50",
) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    output_path = Path(output_path)
    
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw dataset directory does not exist: {raw_dir.resolve()}")
    
    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
    ignore_stems = {"readme", "license", "notes", "description"}
    resample_filter = getattr(Image.Resampling, "LANCZOS", Image.LANCZOS)
    embedder = CNNEmbedder(model_name=cnn_model_name) if use_cnn_embeddings else None

    # Step 1: Discover and unpack images directly from raw_datasets
    temp_work_dir = raw_dir / "_temp_proc_cache"
    temp_work_dir.mkdir(parents=True, exist_ok=True)

    records = []
    
    try:
        # Collect raw image files and unzip PdM_images if present
        all_raw_files = sorted(raw_dir.iterdir())
        
        for file_path in all_raw_files:
            if not file_path.is_file():
                continue
            
            # Unpack image zip files (targeting PdM_images.zip specifically)
            if zipfile.is_zipfile(file_path) and "image" in file_path.name.lower():
                with zipfile.ZipFile(file_path, "r") as z:
                    z.extractall(temp_work_dir)
            elif file_path.suffix.lower() in valid_exts:
                if not any(kw in file_path.stem.lower() for kw in ignore_stems):
                    shutil.copy2(file_path, temp_work_dir / file_path.name)

        # Step 2: Clean, Letterbox, and Extract Features in-memory
        for img_path in sorted(temp_work_dir.rglob("*")):
            if not img_path.is_file() or img_path.suffix.lower() not in valid_exts:
                continue
            if any(kw in img_path.stem.lower() for kw in ignore_stems):
                continue
                
            try:
                with Image.open(img_path) as img:
                    img.verify()
                with Image.open(img_path) as img:
                    img_rgb_pil = img.convert("RGB")
                    
                # Validate variance threshold
                stat = ImageStat.Stat(img_rgb_pil)
                if (sum(stat.stddev) / len(stat.stddev)) < min_std_threshold:
                    continue

                # Resize with aspect ratio preservation (Pad/Letterbox)
                img_padded = ImageOps.pad(img_rgb_pil, target_size, method=resample_filter, color=(0, 0, 0))
                img_rgb = np.array(img_padded)
                gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
                mask = get_padding_mask(img_rgb)

                # Generate features
                record = {
                    "image_filename": f"{img_path.stem}.png",
                    **compute_rust_ratio(img_rgb, mask),
                    **compute_color_stats(img_rgb, mask),
                    **compute_texture_features(gray, mask),
                    **compute_edge_density(gray, mask),
                }

                if embedder is not None:
                    emb = embedder.embed(img_rgb)
                    record.update({f"cnn_emb_{i}": float(v) for i, v in enumerate(emb)})

                records.append(record)
            except Exception:
                continue

    finally:
        shutil.rmtree(temp_work_dir, ignore_errors=True)

    # Step 3: Export to model_train_dataset folder
    df_features = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(output_path, index=False)

    return df_features


# 4. EXECUTION ENTRY POINT

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent if "__file__" in locals() else Path.cwd()
    
    RAW_DIR = BASE_DIR / "datasets" / "raw_datasets"
    FINAL_DATASET_CSV = BASE_DIR / "datasets" / "model_train_dataset" / "image_features.csv"

    # Ingests only from raw_datasets and outputs to model_train_dataset
    df = process_raw_dataset_to_train(
        raw_dir=RAW_DIR,
        output_path=FINAL_DATASET_CSV,
        use_cnn_embeddings=True,
        cnn_model_name="resnet50"
    )

    # Only output printed upon final dataset generation completion
    print(f"[SUCCESS] Model training dataset successfully generated: {FINAL_DATASET_CSV} | Shape: {df.shape}")