import torch
from pathlib import Path

def save_trained_model(model: torch.nn.Module, model_path: Path | str):
    """
    Saves the trained model's state dictionary to the specified path.
    """
    # Convert to Path object if it's a string
    if isinstance(model_path, str):
        model_path = Path(model_path)
        
    # Ensure the parent directory exists
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the model
    torch.save(model.state_dict(), model_path)
    print(f"Model state successfully saved to '{model_path}'")