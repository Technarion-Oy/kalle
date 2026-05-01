import argparse
import torch
import math
import deepspeed
from model import DecoderOnlyTransformer
from config import Config
from tracker import generate_report
from dataset import get_dataloader

def main():
    parser = argparse.ArgumentParser(description="Train custom LLM with DeepSpeed")
    # DeepSpeed local_rank argument
    parser.add_argument("--local_rank", type=int, default=-1, help="local rank passed from distributed launcher")
    # Adds DeepSpeed config arguments (e.g., --deepspeed_config)
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # Initialize DeepSpeed distributed backend
    deepspeed.init_distributed()

    # Initialize the custom Transformer model using centralized config
    config = Config()
    model = DecoderOnlyTransformer(config)

    # Use standard PyTorch Adam optimizer to avoid FusedAdam CUDA requirement on Mac
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.max_lr, weight_decay=config.weight_decay)

    # Initialize DeepSpeed engine
    ds_config_path = args.deepspeed_config if args.deepspeed_config else "ds_config.json"
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        optimizer=optimizer,
        model_parameters=model.parameters(),
        config=ds_config_path
    )

    def get_lr(it, config):
        # 1) linear warmup for warmup_steps steps
        if it < config.warmup_steps:
            return config.max_lr * it / config.warmup_steps
        # 2) if it > lr_decay_iters, return min learning rate
        if it > config.steps_per_epoch:
            return config.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - config.warmup_steps) / (config.steps_per_epoch - config.warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return config.min_lr + coeff * (config.max_lr - config.min_lr)

    # Basic training loop parameters from config
    epochs = config.epochs
    steps_per_epoch = config.steps_per_epoch
    batch_size = model_engine.train_micro_batch_size_per_gpu()
    seq_len = config.max_seq_len

    # Initialize the streaming Dataloader for real TinyStories data
    if model_engine.local_rank <= 0:
        print("Initializing data stream...")
    train_loader = get_dataloader(batch_size, seq_len, split="train")
    train_iter = iter(train_loader)

    device = model_engine.device
    if model_engine.local_rank <= 0:
        print(f"Starting training on {device}...")
    
    global_step = 0
    
    for epoch in range(epochs):
        for step in range(steps_per_epoch):
            # Update learning rate
            lr = get_lr(global_step, config)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # Fetch real data batch
            try:
                x, y = next(train_iter)
            except StopIteration:
                # In streaming, we might hit the end (though TinyStories is huge). Restart if so.
                train_iter = iter(train_loader)
                x, y = next(train_iter)
            
            # Move data to the appropriate device (e.g., MPS, CUDA, or CPU)
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            logits, loss = model_engine(x, targets=y)
            
            # Backward pass 
            model_engine.backward(loss)
            
            # Optimizer step
            model_engine.step()

            global_step += 1
            
            # Logging
            if global_step % config.log_interval == 0 and model_engine.local_rank <= 0:
                print(f"Epoch {epoch} | Step {global_step} | LR: {lr:.2e} | Loss: {loss.item():.4f}")

            # Update metrics tracker
            if global_step % config.tracker_interval == 0:
                if model_engine.local_rank <= 0:
                    print(f"Step {global_step}: Calling tracker to update metrics_report.md...")
                    generate_report()

            # Save checkpoint
            if global_step % config.checkpoint_interval == 0:
                if model_engine.local_rank <= 0:
                    print(f"Step {global_step}: Saving checkpoint...")
                    # Manual torch.save as workaround for DeepSpeed MPS barrier limitation
                    checkpoint_path = f"checkpoint_step_{global_step}.pt"
                    torch.save({
                        'step': global_step,
                        'model_state_dict': model_engine.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss.item(),
                    }, checkpoint_path)
                    print(f"Checkpoint saved to {checkpoint_path}")

    if model_engine.local_rank <= 0:
        print("Training complete.")

if __name__ == "__main__":
    main()
