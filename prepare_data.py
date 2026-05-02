import os
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

def prepare():
    dataset_name = "roneneldan/TinyStories"
    # Use gpt2 encoding to match our Config
    enc = tiktoken.get_encoding("gpt2")
    
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)
    
    for split in ['train', 'validation']:
        print(f"Tokenizing {split} split...")
        data = dataset[split]
        
        # Tokenize everything
        all_tokens = []
        for item in tqdm(data):
            text = item['text']
            tokens = enc.encode_ordinary(text)
            tokens.append(enc.eot_token)
            all_tokens.extend(tokens)
        
        # Convert to numpy array (uint16 is enough for gpt2 vocab size 50257)
        all_tokens_np = np.array(all_tokens, dtype=np.uint16)
        
        # Save as raw binary file
        filename = f"{split}.bin"
        all_tokens_np.tofile(filename)
        print(f"Saved {len(all_tokens)} tokens to {filename}")

if __name__ == "__main__":
    prepare()
