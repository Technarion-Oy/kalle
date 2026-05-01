import math
import torch
import torch.nn as torch_nn
from torch.nn import functional as F
from config import Config

class CausalSelfAttention(torch_nn.Module):
    def __init__(self, args: Config):
        super().__init__()
        assert args.dim % args.n_heads == 0, "Embedding dimension must be divisible by number of heads"
        
        self.n_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads
        
        # Batched linear projection for queries, keys, and values.
        # Math: [B, T, C] @ [C, 3 * C] -> [B, T, 3 * C]
        self.qkv_proj = torch_nn.Linear(args.dim, 3 * args.dim, bias=False)
        
        # Output projection
        # Math: [B, T, C] @ [C, C] -> [B, T, C]
        self.o_proj = torch_nn.Linear(args.dim, args.dim, bias=False)
        self.dropout = torch_nn.Dropout(args.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size() # B: Batch size, T: Sequence length, C: Embedding dimension (args.dim)

        # 1. Project to Q, K, V
        qkv = self.qkv_proj(x)
        
        # 2. Split into Q, K, V and reshape for multi-head attention
        # Reshape: [B, T, 3 * C] -> [B, T, 3, n_heads, head_dim] -> 3 x [B, n_heads, T, head_dim]
        qkv = qkv.view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        
        # Transpose spatial dimensions to match sequence operations: [B, n_heads, T, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 3. Flash Attention via PyTorch's native scaled_dot_product_attention.
        # This automatically dispatches to highly optimized FlashAttention kernels in PyTorch 2.0+.
        # Math: Attention(Q, K, V) = softmax((Q @ K^T) / sqrt(d_k)) @ V
        # The is_causal=True flag applies a lower-triangular boolean mask matrix to prevent attending to future tokens.
        y = F.scaled_dot_product_attention(
            q, k, v, 
            dropout_p=self.dropout.p if self.training else 0.0, 
            is_causal=True
        )

        # 4. Concatenate heads and project output
        # Reshape: [B, n_heads, T, head_dim] -> [B, T, n_heads, head_dim] -> [B, T, C]
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.o_proj(y)

class FeedForward(torch_nn.Module):
    def __init__(self, args: Config):
        super().__init__()
        # Standard MLP expansion factor is 4.
        # Math: [B, T, C] @ [C, 4 * C] -> [B, T, 4 * C]
        self.fc1 = torch_nn.Linear(args.dim, 4 * args.dim, bias=False)
        self.fc2 = torch_nn.Linear(4 * args.dim, args.dim, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Math: x = xW_1
        #       x = GELU(x) = x * 0.5 * (1.0 + erf(x / sqrt(2.0)))
        #       x = xW_2
        return self.fc2(F.gelu(self.fc1(x)))

class TransformerBlock(torch_nn.Module):
    def __init__(self, args: Config):
        super().__init__()
        # Layer Normalization: y = (x - E[x]) / sqrt(Var[x] + eps) * gamma + beta
        self.ln_1 = torch_nn.LayerNorm(args.dim)
        self.attn = CausalSelfAttention(args)
        self.ln_2 = torch_nn.LayerNorm(args.dim)
        self.mlp = FeedForward(args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual connections around Attention and MLP.
        # Math: x = x + Attention(LayerNorm(x))
        x = x + self.attn(self.ln_1(x))
        # Math: x = x + MLP(LayerNorm(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class DecoderOnlyTransformer(torch_nn.Module):
    def __init__(self, args: Config):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        
        # Token Embedding: maps token integers to vectors of size `dim`.
        self.token_emb = torch_nn.Embedding(args.vocab_size, args.dim)
        # Positional Embedding: maps positions [0, max_seq_len-1] to vectors of size `dim`.
        self.pos_emb = torch_nn.Embedding(args.max_seq_len, args.dim)
        self.dropout = torch_nn.Dropout(args.dropout)

        # Stack of Transformer blocks
        self.blocks = torch_nn.ModuleList([TransformerBlock(args) for _ in range(args.n_layers)])
        
        # Final LayerNorm before the output projection.
        self.ln_f = torch_nn.LayerNorm(args.dim)
        
        # Language Modeling Head: projects the hidden states back to the vocabulary size.
        # Math: [B, T, C] @ [C, Vocab_Size] -> [B, T, Vocab_Size]
        self.lm_head = torch_nn.Linear(args.dim, args.vocab_size, bias=False)

        # Weight tying: share weights between token embedding and the language modeling head.
        self.token_emb.weight = self.lm_head.weight
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # Standard GPT weight initialization.
        if isinstance(module, torch_nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch_nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        B, T = idx.size()
        assert T <= self.args.max_seq_len, f"Cannot forward sequence of length {T}, block size is only {self.args.max_seq_len}"

        # 1. Embeddings
        # Extract token and position embeddings, then add them.
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        
        tok_emb = self.token_emb(idx) # [B, T, C]
        pos_emb = self.pos_emb(pos)   # [T, C]
        
        x = self.dropout(tok_emb + pos_emb)

        # 2. Transformer Blocks
        for block in self.blocks:
            x = block(x)

        # 3. Final normalization and output projection
        x = self.ln_f(x)
        logits = self.lm_head(x) # [B, T, Vocab_Size]

        loss = None
        if targets is not None:
            # CrossEntropyLoss expects logits of shape [B * T, Vocab_Size] and targets [B * T]
            # Math: Loss = -log(exp(logits[target]) / sum(exp(logits)))
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = targets.view(-1)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss
