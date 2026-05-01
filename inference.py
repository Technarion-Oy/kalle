import torch
import torch.nn.functional as F
import tiktoken
from model import DecoderOnlyTransformer
from config import Config
import argparse
import os

def generate(model, prompt, max_new_tokens=100, temperature=0.8, top_k=50, device="cpu"):
    """
    Autoregressively generates tokens from a prompt.
    """
    model.eval()
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Encode prompt to tokens
    idx = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    
    print(f"--- Generating (Prompt: '{prompt}') ---")
    print(prompt, end="", flush=True)

    for _ in range(max_new_tokens):
        # Crop the context if it exceeds max_seq_len
        idx_cond = idx if idx.size(1) <= model.args.max_seq_len else idx[:, -model.args.max_seq_len:]
        
        # Forward pass to get logits for the last token
        logits, _ = model(idx_cond)
        
        # Pluck the logits of the last time step and scale by temperature
        logits = logits[:, -1, :] / temperature
        
        # Optional: Top-K sampling
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
            
        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Sample from the distribution
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Append to the sequence
        idx = torch.cat((idx, next_token), dim=1)
        
        # Decode and print the single token
        token_str = tokenizer.decode([next_token.item()])
        print(token_str, end="", flush=True)
        
        # Stop if EOT token is generated
        if next_token.item() == tokenizer.eot_token:
            break
            
    print("\n--- End of Generation ---")

def main():
    parser = argparse.ArgumentParser(description="Generate text from custom LLM")
    parser.add_argument("--checkpoint", type=str, default="checkpoint_step_5000.pt", help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, default="Once upon a time, there was a little dog named", help="Prompt for generation")
    parser.add_argument("--tokens", type=int, default=100, help="Number of tokens to generate")
    parser.add_argument("--temp", type=float, default=0.8, help="Sampling temperature")
    args = parser.parse_args()

    config = Config()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Initialize model
    model = DecoderOnlyTransformer(config)
    
    # Load checkpoint
    if os.path.exists(args.checkpoint):
        print(f"Loading weights from {args.checkpoint}...")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict)
    else:
        print(f"Error: Checkpoint {args.checkpoint} not found.")
        return
        
    model.to(device)
    
    generate(model, args.prompt, max_new_tokens=args.tokens, temperature=args.temp, device=device)

if __name__ == "__main__":
    main()
