import torch
from torch.utils.data import IterableDataset, DataLoader
import tiktoken
from datasets import load_dataset

class StreamingTokenDataset(IterableDataset):
    def __init__(self, seq_len: int, split="train", buffer_size=100):
        super().__init__()
        self.seq_len = seq_len
        
        # Stream the TinyStories dataset to avoid large RAM footprint.
        # This streams over network instead of downloading everything.
        self.dataset = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
        
        # Limit the shuffle buffer to prevent RAM bloat on local machine
        self.dataset = self.dataset.shuffle(seed=42, buffer_size=buffer_size)
        
        # Use gpt2 tokenizer which has exactly 50257 vocab size, matching our ModelArgs
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.eot = self.tokenizer.eot_token

    def __iter__(self):
        buffer = []
        for item in self.dataset:
            text = item["text"]
            # Encode the story and add End-Of-Text token
            tokens = self.tokenizer.encode(text, allowed_special="all")
            buffer.extend(tokens)
            buffer.append(self.eot)
            
            # Yield full sequences of size seq_len + 1 (for inputs and targets)
            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[:self.seq_len + 1]
                # Advance buffer by seq_len (predicting the next token across chunks seamlessly)
                buffer = buffer[self.seq_len:] 
                
                # x: [token_0, ..., token_seq_len-1]
                # y: [token_1, ..., token_seq_len]
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                yield x, y

def get_dataloader(batch_size: int, seq_len: int, split="train"):
    """
    Returns a PyTorch DataLoader that streams tokenized TinyStories chunks.
    """
    dataset = StreamingTokenDataset(seq_len=seq_len, split=split)
    # num_workers=0 is ideal for basic streaming IterableDatasets without complex sharding logic
    return DataLoader(dataset, batch_size=batch_size, num_workers=0)
