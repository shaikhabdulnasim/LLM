# LLM from Scratch: PyTorch Implementation

## Overview
This repository contains a complete, step-by-step implementation of a Large Language Model (LLM) built from scratch using PyTorch. Based on the architectural principles of GPT-style models, this project demonstrates the end-to-end pipeline of building, pretraining, and fine-tuning a Transformer network without relying on high-level abstraction libraries.

The repository tracks the progression from basic tensor operations and attention mechanics up to a fully assembled GPT model capable of instruction fine-tuning.

## File Directory & Descriptions

Below is a guide to the files in this repository, breaking down the core engine scripts, Jupyter notebooks used for experimentation, and intermediate learning steps.

### Core Engine (Python Scripts)
*   **`attention.py`**: The finalized implementation of the attention mechanisms, including Scaled Dot-Product, Causal Masking, and Multi-Head Attention.
*   **`gptDummy.py`**: The main Transformer architecture. It assembles the embeddings, transformer blocks (LayerNorm, GELU, Feed-Forward), and final output heads.
*   **`data_loader.py`**: Custom PyTorch `Dataset` and `DataLoader` classes to handle sliding-window context and efficient tensor batching.
*   **`tokenization.py`**: The Byte-Pair Encoding (BPE) tokenizer used to convert raw text into integer vocabularies.
*   **`training.py`**: Contains the core training loops, optimization steps (AdamW), and loss calculations.
*   **`pretrain_generate.py`**: The execution script for autoregressive pretraining (next-token prediction) and text generation functions (like temperature scaling and top-k decoding).

### Experimentation & Fine-Tuning (Jupyter Notebooks)
*   **`instruction-Finetuning.ipynb`**: Demonstrates Supervised Fine-Tuning (SFT). Covers formatting instruction prompts, dynamic padding, and applying cross-entropy loss masking (`ignore_index=-100`).
*   **`classification-finetuning.ipynb`**: Demonstrates how to adapt the pretrained LLM for text classification tasks.
*   **`gptDummy.ipynb` & `gpt_block.ipynb`**: Step-by-step walkthroughs of building the individual Transformer blocks and the overarching GPT architecture.
*   **`pretrain_generate.ipynb`**: A visual, step-by-step notebook showing the model learning to generate text during the pretraining phase.
*   **`Self_attention_compact_class.ipynb`**: Notebook experimenting with the mathematical operations behind self-attention.

### Intermediate Learning Files & Data
*   **`Self_atten_trainableweighs.py`**: Intermediate script demonstrating how attention weights are initialized and updated.
*   **`self_atten_without_trainableweig...py`**: Early experimental script showing raw attention mechanics before introducing trainable weight matrices.
*   **`the-verdict.txt`**: The raw text dataset used to train and test the data loading and pretraining pipelines.

## How to Run

To test the main architecture and text generation, ensure you have PyTorch installed and run:

```bash
python pretrain_generate.py
