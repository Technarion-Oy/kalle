import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class MemmapDataset(Dataset):
    def __init__(self, filename, seq_len):
        super().__init__()
        self.filename = filename
        self.seq_len = seq_len
        
        # Open the raw binary file using memmap for O(1) RAM usage regardless of file size
        self.data = np.memmap(filename, dtype=np.uint16, mode='r')
        
        # Number of samples is total tokens divided by seq_len
        # We use a simple indexing strategy: each sample is seq_len tokens
        # and we slide by seq_len (non-overlapping for simplicity in this refactor)
        self.n_samples = (len(self.data) - 1) // seq_len

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start_idx = idx * self.seq_len
        end_idx = start_idx + self.seq_len + 1
        
        # Extract the chunk
        chunk = self.data[start_idx:end_idx].astype(np.int64)
        
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        
        return x, y

def get_dataloader(batch_size: int, seq_len: int, split="train"):
    filename = f"{split}.bin"
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Binary file {filename} not found. Run prepare_data.py first.")
        
    dataset = MemmapDataset(filename, seq_len)
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=(split == "train"), 
        num_workers=0, # memmap is thread-safe but works best in main process for simple scripts
        pin_memory=False # MPS doesn't benefit from CPU-side pinning like CUDA
    )
