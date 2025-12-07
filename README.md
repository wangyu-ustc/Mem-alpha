# Mem-α: Learning Memory Construction via Reinforcement Learning

[![arXiv](https://img.shields.io/badge/arXiv-2509.25911-b31b1b.svg)](https://arxiv.org/abs/2509.25911)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Memalpha--4B-yellow)](https://huggingface.co/YuWangX/Memalpha-4B)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Official implementation of **"Mem-α: Learning Memory Construction via Reinforcement Learning"**. 

## Table of Contents

- [Mem-α: Learning Memory Construction via Reinforcement Learning](#mem-α-learning-memory-construction-via-reinforcement-learning)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Installation](#installation)
  - [Dataset Preparation](#dataset-preparation)
    - [Prerequisites: Install Git LFS](#prerequisites-install-git-lfs)
    - [Download Datasets](#download-datasets)
  - [Training](#training)
    - [Main Model](#main-model)
    - [Ablation Studies](#ablation-studies)
  - [Evaluation](#evaluation)
    - [Running the Memory Server](#running-the-memory-server)
    - [Evaluating Trained Models](#evaluating-trained-models)
    - [Baseline Evaluations](#baseline-evaluations)
      - [Long-Context, RAG Baselines and MemAgent](#long-context-rag-baselines-and-memagent)
      - [MEM1 Baseline](#mem1-baseline)
  - [Dataset Processing](#dataset-processing)
    - [Memalpha Dataset](#memalpha-dataset)
    - [MemoryAgentBench Dataset](#memoryagentbench-dataset)
  - [Citation](#citation)
  - [License](#license)

## Overview

Large language model (LLM) agents are constrained by limited context windows, necessitating external memory systems for long-term information understanding. Mem-α is a reinforcement learning framework that trains agents to effectively manage complex memory systems through interaction and feedback.

**Key Features:**
- 🧠 **Advanced Memory Architecture**: Core, episodic, and semantic memory components
- 🎯 **Reinforcement Learning Framework**: Direct optimization for memory construction
- 📈 **Strong Generalization**: Trained on 30k tokens, generalizes to 400k+ tokens (13x training length)
- 🚀 **State-of-the-art Performance**: Significant improvements over existing memory-augmented agents

**Resources:**
- 📄 [Paper](https://arxiv.org/abs/2509.25911)
- 🤗 [Model (Memalpha-4B)](https://huggingface.co/YuWangX/Memalpha-4B)
- 📊 [Training Dataset](https://huggingface.co/datasets/YuWangX/Memalpha)
- 📊 [Evaluation Dataset - Processed Version](https://huggingface.co/datasets/YuWangX/Memalpha-Memoryagentbench)
- 📊 [MemoryAgentBench - Original](https://huggingface.co/datasets/ai-hyz/MemoryAgentBench)

## Installation

```bash
# Clone the repository
git clone git@github.com:wangyu-ustc/Mem-alpha.git
cd Mem-alpha

# Install dependencies
pip install -r requirements.txt
```

## Dataset Preparation

### Prerequisites: Install Git LFS

The datasets are stored using Git Large File Storage (LFS). Before downloading, you need to install Git LFS:

```bash
# macOS
brew install git-lfs

# Ubuntu/Debian
sudo apt-get install git-lfs

# Install for your user account
git lfs install
```

### Download Datasets

Create a `data` folder in the project root and download the datasets:

```bash
# Download Memalpha training/test dataset
git clone https://huggingface.co/datasets/YuWangX/Memalpha ./data/memalpha
cd ./data/memalpha
git lfs pull  # Pull the actual dataset files (not just LFS pointers)
cd ../..

# Download MemoryAgentBench evaluation dataset (processed version for this project)
git clone https://huggingface.co/datasets/YuWangX/Memalpha-Memoryagentbench ./data/memoryagentbench
cd ./data/memoryagentbench
git lfs pull  # Pull the actual dataset files
cd ../..
```

> **⚠️ Important:** Without Git LFS installed, you'll only download small pointer files (~133 bytes) instead of the actual datasets (~62 MB for memalpha). Make sure to run `git lfs pull` after cloning to download the real data files.

> **Note:** We use a processed version of the [original MemoryAgentBench](https://huggingface.co/datasets/ai-hyz/MemoryAgentBench) dataset. See [Dataset Processing](#dataset-processing) for details.

**Expected directory structure:**
```
data/
├── memalpha/
│   ├── train.parquet
│   └── test.parquet
└── memoryagentbench/
    ├── train.parquet
    └── test.parquet
```

> **Note:** If you prefer to process the datasets from scratch, see [Dataset Processing](#dataset-processing).

## Training

### Main Model

To train the Memalpha-4B model with optimal hyperparameters (β=0.05, γ=0.1):

```bash
bash scripts/train_memory_grpo_qwen3-4b-4node-compression0.05-content0.1.sh
```

### Ablation Studies

The following scripts reproduce the ablation study results from the paper:

```bash
# β=0.05, γ=0.0 (No content reward)
bash scripts/train_memory_grpo_qwen3-4b-4node-compression0.05-content0.0.sh

# β=0.0, γ=0.1 (No compression reward)
bash scripts/train_memory_grpo_qwen3-4b-4node-compression0.0-content0.1.sh

# β=0.05, γ=0.1 (Main configuration)
bash scripts/train_memory_grpo_qwen3-4b-4node-compression0.05-content0.1.sh

# β=0.2, γ=0.1 (Higher compression penalty)
bash scripts/train_memory_grpo_qwen3-8b-4node-compression0.2-content0.1.sh

# β=0.4, γ=0.1 (Highest compression penalty)
bash scripts/train_memory_grpo_qwen3-4b-4node-compression0.4-content0.1.sh
```

**Parameter explanations:**
- **β (beta)**: Compression reward coefficient - penalizes excessive memory usage
- **γ (gamma)**: Content reward coefficient - measure whether the information is put into the correct memory type

## Evaluation

### Running the Memory Server

1. If you deploy Qwen3-32B using vllm without any API_KEY, then you simply need to set the following environmental variable:
```bash
QWEN_URL="http://localhost:8001/v1" # replace with your qwen url. 
QWEN_MODEL_NAME="qwen3-32b" # replace with your qwen model. 
``` 

If you want to use `openrouter`, then configure your `.env` (or shell) with the OpenRouter credentials. You only need the API key plus the Qwen endpoint/model:

```bash
OPENROUTER_API_KEY=sk-...
QWEN_URL="https://openrouter.ai/api/v1"      # copy the base URL from OpenRouter
QWEN_MODEL_NAME="qwen/qwen3-32b"             # optional override (defaults to qwen3-32b)
```

2. Start the memory server (no extra flags needed unless you want to override the base URL for a custom endpoint):

```bash
python memory_server.py --port 5005
```

### Evaluating Trained Models

Evaluate the trained Memalpha-4B model on both datasets:

```bash
# Evaluate on Memalpha dataset
python main.py --agent_config config/memalpha-qwen3-4b_agent_0.05-0.1.yaml --dataset memalpha

# Evaluate on MemoryAgentBench dataset
python main.py --agent_config config/memalpha-qwen3-4b_agent_0.05-0.1.yaml --dataset memoryagentbench
```

### Baseline Evaluations

We provide evaluation scripts for several baseline methods:

#### Long-Context, RAG Baselines and MemAgent

Evaluate long-context models and BM25-based retrieval on both datasets:

```bash
# Memalpha dataset
python long_context_eval.py --model qwen3-32b --dataset memalpha              # Long-context baseline
python long_context_eval.py --model qwen3-32b-bm25 --dataset memalpha         # BM25 retrieval
python long_context_eval.py --model gpt-4o-mini --dataset memalpha            # GPT-4o-mini
python long_context_eval.py --model memagent-14b --dataset memalpha           # MemAgent baseline

# MemoryAgentBench dataset
python long_context_eval.py --model qwen3-32b --dataset memoryagentbench
python long_context_eval.py --model qwen3-32b-bm25 --dataset memoryagentbench
python long_context_eval.py --model gpt-4o-mini --dataset memoryagentbench
python long_context_eval.py --model memagent-14b --dataset memoryagentbench
```

#### MEM1 Baseline

To evaluate [MEM1](https://github.com/MIT-MI/MEM1) as a baseline:

```bash
# Start the VLLM server for MEM1
cd MEM1/Mem1/inference
bash start_vllm.sh
cd ../../..

# Run MEM1 evaluation on both datasets
cd MEM1/Mem1
python inference/generate_rollout.py \
    --model Mem-Lab/Qwen2.5-7B-RL-RAG-Q2-EM-Release \
    --use_mem1 \
    --data_file ../../data/memalpha/test.parquet

python inference/generate_rollout.py \
    --model Mem-Lab/Qwen2.5-7B-RL-RAG-Q2-EM-Release \
    --use_mem1 \
    --data_file ../../data/memoryagentbench/test.parquet

# For the results reported in this repo, we load the MEM1 rollouts via our evaluator by setting the model to `mem1`:
# (requires `MEM1/Mem1/<dataset>_results.json` produced by the commands above)
python long_context_eval.py --model mem1 --dataset memalpha
python long_context_eval.py --model mem1 --dataset memoryagentbench
```

## Dataset Processing

### Memalpha Dataset

If you want to build the Memalpha dataset from scratch instead of downloading it:

```bash
# Process individual datasets
python process_data.py --dataset squad
python process_data.py --dataset squad --split-single-dataset

python process_data.py --dataset hotpotqa
python process_data.py --dataset hotpotqa --split-single-dataset

python process_data.py --dataset booksum
python data_preprocess/extract_booksum_keywords.py --mode replace
python process_data.py --dataset booksum --split-single-dataset

python process_data.py --dataset pubmed-rct
python process_data.py --dataset pubmed-rct --split-single-dataset

python process_data.py --dataset perltqa
python process_data.py --dataset perltqa --split-single-dataset

python process_data.py --dataset ttl_train
python process_data.py --dataset ttl_train --split-single-dataset

python process_data.py --dataset lme_train
python process_data.py --dataset lme_train --split-single-dataset

# Merge all datasets
python process_data.py --merge-datasets pubmed-rct lme_train squad hotpotqa perltqa ttl_train booksum --limit-size 100
```

### MemoryAgentBench Dataset

To build the MemoryAgentBench evaluation dataset from scratch:

```bash
# Process MemoryAgentBench components
python process_data.py --dataset accurate_retrieval
python process_data.py --dataset test_time_learning
python process_data.py --dataset long_range_understanding

# Merge into final evaluation set
python process_data.py --merge-datasets accurate_retrieval long_range_understanding test_time_learning --output-name memoryagentbench
```

> **⚠️ Warning:** Since [MemoryAgentBench](https://huggingface.co/datasets/ai-hyz/MemoryAgentBench) is continuously updated, processing from scratch may yield different results than the published dataset. We recommend downloading our processed version directly from [HuggingFace](https://huggingface.co/datasets/YuWangX/Memalpha-Memoryagentbench) for reproducibility.
> 
> **Note:** Our evaluation uses a processed version of the original [MemoryAgentBench dataset](https://huggingface.co/datasets/ai-hyz/MemoryAgentBench) ([paper](https://arxiv.org/abs/2507.05257)). The processing scripts above show how we adapted it for our experiments.

## Citation

If you find this work helpful, please cite our paper:

```bibtex
@article{wang2025memalpha,
  title={Mem-$\alpha$: Learning Memory Construction via Reinforcement Learning},
  author={Wang, Yu and Takanobu, Ryuichi and Liang, Zhiqi and Mao, Yuzhen and Hu, Yuanzhe and McAuley, Julian and Wu, Xiaojian},
  journal={arXiv preprint arXiv:2509.25911},
  year={2025}
}
```

If you use our processed MemoryAgentBench dataset, please also cite the original work:

```bibtex
@article{hu2025evaluating,
  title={Evaluating memory in llm agents via incremental multi-turn interactions},
  author={Hu, Yuanzhe and Wang, Yu and McAuley, Julian},
  journal={arXiv preprint arXiv:2507.05257},
  year={2025}
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

**Contact:** For questions or issues, please open an issue on GitHub or contact Yu Wang at [yuw164@ucsd.edu](mailto:yuw164@ucsd.edu).
