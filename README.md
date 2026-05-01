# LLM Optimization from Scratch

A high-performance implementation of a decoder-only Transformer optimized for training on Apple Silicon (MPS) using DeepSpeed and PyTorch.

## Features
*   **Custom Transformer**: Decoder-only architecture with FlashAttention (via PyTorch native SDPA).
*   **DeepSpeed Integration**: Optimized for distributed training and ZeRO stages.
*   **Apple Silicon Optimized**: Specifically tuned to handle 16GB Unified Memory constraints via aggressive gradient accumulation and streaming datasets.
*   **Real-time Monitoring**: Custom `tracker.py` to monitor system RAM, CPU, and Disk I/O.
*   **Streaming Data**: Integrated with Hugging Face `datasets` and `tiktoken` for low-memory data pipelines.

## Project Structure
*   `model.py`: Core Transformer architecture.
*   `train.py`: Main training loop with DeepSpeed and LR scheduling.
*   `dataset.py`: Streaming data pipeline using `TinyStories`.
*   `config.py`: Centralized hyperparameter management.
*   `eval.py`: Validation loss and perplexity evaluation.
*   `inference.py`: Autoregressive text generation.
*   `tracker.py`: Performance monitoring utility.

## Setup
1.  **Clone the repository**:
    ```bash
    git clone https://github.com/Technarion-Oy/kalle.git
    cd kalle
    ```
2.  **Create virtual environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training
To start or resume training using the DeepSpeed launcher:
```bash
deepspeed train.py
```

### Evaluation
To evaluate a specific checkpoint:
```bash
python eval.py --checkpoint checkpoint_step_5000.pt
```

### Inference
To generate stories from the model:
```bash
python inference.py --prompt "Once upon a time, there was a little dog named"
```

## Performance Monitoring
The `metrics_report.md` file is automatically updated every 100 steps during training to track system resource usage.
