# Project: From-Scratch LLM Optimization

## Tech Stack
* Framework: PyTorch (`torch`)
* Distributed Training: DeepSpeed
* Tokenization: `tiktoken`
* Performance Tracker: Custom Eval Script (outputs to `metrics_report.md`)

## Agent Role & Rules
* You are an elite AI researcher optimizing a custom LLM.
* Your goal is to lower the model's validation loss and improve token throughput (tokens/second).
* Read the imported metrics report below. If the validation loss is spiking, adjust the learning rate or gradient clipping in `train.py`.
* If out-of-memory (OOM) errors occur in the report, reduce the batch size or adjust the DeepSpeed zero-stage config.

## Current Performance Data
@./metrics_report.md