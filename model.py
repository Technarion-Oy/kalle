import math
import torch
import torch.nn as torch_nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from config import Config

class CausalSelfAttention(torch_nn.Module):
    def __init__(self, args: Config):
        super().__init__()
        assert args.dim % args.n_heads == 0, "Embedding dimension must be divisible by number of heads"
        
        self.n_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads
        
        # Batched linear projection for queries, keys, and values.
        self.qkv_proj = torch_nn.Linear(args.dim, 3 * args.dim, bias=False)
        
        # Output projection
        self.o_proj = torch_nn.Linear(args.dim, args.dim, bias=False)
        self.dropout = torch_nn.Dropout(args.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        # 1. Project and reshape
        qkv = self.qkv_proj(x)
        qkv = qkv.view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        
        # Transpose for SDPA: [B, n_heads, T, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 2. Optimized MPS Execution via Scaled Dot Product Attention
        # PyTorch dispatches this to optimized kernels for Apple Silicon.
        y = F.scaled_dot_product_attention(
            q, k, v, 
            dropout_p=self.dropout.p if self.training else 0.0, 
            is_causal=True
        )

        # 3. Concatenate and project
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y)

class FeedForward(torch_nn.Module):
    def __init__(self, args: Config):
        super().__init__()
        self.fc1 = torch_nn.Linear(args.dim, 4 * args.dim, bias=False)
        self.fc2 = torch_nn.Linear(4 * args.dim, args.dim, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))

class TransformerBlock(torch_nn.Module):
    def __init__(self, args: Config):
        super().__init__()
        self.ln_1 = torch_nn.LayerNorm(args.dim)
        self.attn = CausalSelfAttention(args)
        self.ln_2 = torch_nn.LayerNorm(args.dim)
        self.mlp = FeedForward(args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual connections
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class DecoderOnlyTransformer(torch_nn.Module):
    def __init__(self, args: Config):
        super().__init__()
        self.args = args
        self.gradient_checkpointing = False # Enabled via train.py
        
        self.token_emb = torch_nn.Embedding(args.vocab_size, args.dim)
        self.pos_emb = torch_nn.Embedding(args.max_seq_len, args.dim)
        self.dropout = torch_nn.Dropout(args.dropout)

        self.blocks = torch_nn.ModuleList([TransformerBlock(args) for _ in range(args.n_layers)])
        self.ln_f = torch_nn.LayerNorm(args.dim)
        self.lm_head = torch_nn.Linear(args.dim, args.vocab_size, bias=False)

        self.token_emb.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch_nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch_nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        B, T = idx.size()
        
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.dropout(self.token_emb(idx) + self.pos_emb(pos))

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                # Gradient Checkpointing: trade compute for VRAM
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss
