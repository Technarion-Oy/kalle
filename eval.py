import torch
from model import DecoderOnlyTransformer
from config import Config
from dataset import get_dataloader
import argparse
import os

def evaluate(model, dataloader, device, max_steps=100):
    model.eval()
    total_loss = 0
    steps = 0
    
    print(f"Evaluating for {max_steps} steps...")
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            _, loss = model(x, targets=y)
            total_loss += loss.item()
            steps += 1
            
            if steps >= max_steps:
                break
                
    avg_loss = total_loss / steps if steps > 0 else 0
    return avg_loss

def main():
    parser = argparse.ArgumentParser(description="Evaluate custom LLM")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--steps", type=int, default=50, help="Number of evaluation steps")
    args = parser.parse_args()

    config = Config()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Initialize model
    model = DecoderOnlyTransformer(config)
    model.to(device)
    
    # Load checkpoint
    if os.path.exists(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}...")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        # Handle cases where model_state_dict is nested or direct
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict)
    else:
        print(f"Error: Checkpoint {args.checkpoint} not found.")
        return

    # Initialize validation data stream
    val_loader = get_dataloader(config.train_batch_size, config.max_seq_len, split="validation")
    
    val_loss = evaluate(model, val_loader, device, max_steps=args.steps)
    
    print(f"--- Evaluation Results ---")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Perplexity: {torch.exp(torch.tensor(val_loss)).item():.4f}")

if __name__ == "__main__":
    main()
