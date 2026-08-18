import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import auc, precision_recall_curve, precision_score, recall_score

# Import model architecture and dataset class from training script
from model_training import MultimodalMaintenanceDataset, StreamingMultimodalTransformer

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 1. EVALUATION & THRESHOLD TUNING
# ==========================================
def evaluate_and_tune(model, dataset):
    model.eval()
    val_loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    all_targets = []
    all_probs = []
    all_ttfs = []

    with torch.no_grad():
        for batch in val_loader:
            ts = batch["ts"].to(device)
            num = batch["num"].to(device)
            img = batch["img"].to(device)
            
            logits, ttf_pred = model(ts, num, img)
            probs = torch.sigmoid(logits)
            
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(batch["label"].numpy())
            all_ttfs.extend(ttf_pred.cpu().numpy())

    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    all_ttfs = np.array(all_ttfs)

    # Precision-Recall Curve & PR-AUC
    precision, recall, thresholds = precision_recall_curve(all_targets, all_probs)
    pr_auc = auc(recall, precision)

    # Threshold Tuning for Max F1-Score
    f1_scores = [
        2 * (p * r) / (p + r + 1e-10) for p, r in zip(precision, recall)
    ]
    best_idx = np.argmax(f1_scores)
    optimal_threshold = (
        thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    )
    best_f1 = f1_scores[best_idx]

    # Metrics at tuned threshold
    predictions = (all_probs >= optimal_threshold).astype(int)
    tuned_precision = precision_score(all_targets, predictions, zero_division=0)
    tuned_recall = recall_score(all_targets, predictions, zero_division=0)

    print("\n" + "=" * 45)
    print("        EVALUATION & THRESHOLD RESULTS       ")
    print("=" * 45)
    print(f"PR-AUC (Average Precision) : {pr_auc:.4f}")
    print(f"Optimal Probability Threshold: {optimal_threshold:.4f}")
    print(f"F1-Score at Threshold      : {best_f1:.4f}")
    print(f"Precision at Threshold     : {tuned_precision:.4f}")
    print(f"Recall at Threshold        : {tuned_recall:.4f}")
    print("=" * 45)

    return optimal_threshold, all_probs, all_ttfs


# ==========================================
# 2. SINGLE INFERENCE DEMO
# ==========================================
def run_sample_inference(model, dataset, threshold):
    random_idx = np.random.randint(0, len(dataset))
    sample = dataset[random_idx]

    ts_data = sample["ts"].unsqueeze(0).to(device)
    num_data = sample["num"].unsqueeze(0).to(device)
    img_data = sample["img"].unsqueeze(0).to(device)
    true_label = int(sample["label"].item())
    true_ttf = sample["ttf"].item()

    model.eval()
    with torch.no_grad():
        logit, ttf_pred = model(ts_data, num_data, img_data)
        prob = torch.sigmoid(logit).item()
        pred_ttf = ttf_pred.item()

    alert_status = "ALERT: FAILURE IMMINENT" if prob >= threshold else "NORMAL"

    print("\n" + "=" * 55)
    print(f"         RANDOM VALIDATION ROW TEST (Index: {random_idx})")
    print("=" * 55)
    print("\n--- INPUT DATA FEATURES ---")
    print(f"Numerical Features (Age/Specs) : {num_data.squeeze(0).cpu().numpy().tolist()}")
    print(
        f"Time-Series Shape (Sensors)    : {ts_data.squeeze(0).shape}"
        f" (Last step: {ts_data[0, -1].cpu().numpy().round(2).tolist()})"
    )
    print(f"Image Tensor Shape             : {img_data.squeeze(0).shape}")
    print("\n--- GROUND TRUTH ---")
    print(f"Actual Failure Label           : {true_label}")
    print(f"Actual Time-To-Failure (TTF)   : {true_ttf:.2f} Hours")
    print("\n--- MODEL PREDICTION ---")
    print(f"Predicted Failure Probability  : {prob * 100:.2f}%")
    print(f"Tuned Failure Alert Binary     : {alert_status}")
    print(f"Predicted Time-To-Failure (TTF): {pred_ttf:.2f} Hours")
    print("=" * 55)


# Execution Workflow
if __name__ == "__main__":
    model_path = "model.pt"

    # Load Model Checkpoint
    model = StreamingMultimodalTransformer().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded weights successfully from '{model_path}'")

    # Generate Validation Data
    val_dataset = MultimodalMaintenanceDataset(
        num_samples=300, sequence_length=64, failure_rate=0.08
    )

    # Evaluate & Sample
    optimal_threshold, _, _ = evaluate_and_tune(model, val_dataset)
    run_sample_inference(model, val_dataset, optimal_threshold)