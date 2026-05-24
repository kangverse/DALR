# DALR

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.7+](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://www.python.org/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

[English](README.md) | [中文](README_zh.md)

## Overview

We propose **DALR** (**D**ual-level **A**lignment **L**earning for multimodal sentence **R**epresentation Learning).

To achieve cross-modal fine-grained alignment, we propose a cross-modal alignment method to mitigate the *cross-modal misalignment bias* (CMB) issue. To alleviate the *intra-modal semantic divergence* (ISD) issue, we integrate ranking distillation with global alignment learning to effectively align intra-modal representations.

The figure below illustrates the overall model architecture.

![DALR model architecture](figure/model.png)

### Paper Links

- arXiv: https://arxiv.org/abs/2506.21096
- ACL Anthology (Findings ACL 2025): https://aclanthology.org/2025.findings-acl.183/

---

## Table of Contents

- [Getting Started](#getting-started)
  - [Environment Setup](#environment-setup)
  - [Download Datasets](#download-datasets)
- [Quick Start: Use DALR](#quick-start-use-dalr)
- [Evaluation](#evaluation)
- [Train Your Own Models](#train-your-own-models)
- [FAQ](#faq)
- [Troubleshooting](#troubleshooting)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [Contributing](#contributing)

---

## Getting Started

### Environment Setup

We recommend creating a virtual environment first:

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

Install PyTorch (CUDA 11.1):

```bash
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 \
    -f https://download.pytorch.org/whl/torch_stable.html
```

For CUDA < 11 or CPU-only:

```bash
pip install torch==1.8.1
```

Then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

### Download Datasets

Download Flickr30k and MS-COCO from their official websites and organize them as follows:

```
REPO ROOT
├── data
│   ├── Flickr/
│   ├── MS-COCO/
│   └── wiki1m_for_simcse.txt
├── Model/
│   ├── bert-base-uncased/
│   ├── simcse/
│   ├── DiffCSE/
│   └── clip/
│       └── ViT-B-32.pt
```

**Wiki1M** (used for text training):

```bash
wget https://huggingface.co/datasets/princeton-nlp/datasets-for-simcse/resolve/main/wiki1m_for_simcse.txt \
    -P data/
```

**SentEval evaluation datasets** (from [SimCSE](https://github.com/princeton-nlp/SimCSE)):

```bash
cd SentEval/data/downstream/
bash download_dataset.sh
```

Pretrained models (SimCSE, DiffCSE, BERT-base, CLIP ViT-B/32) can be downloaded from [Hugging Face](https://huggingface.co/) and placed in the `Model/` directory.

---

## Quick Start: Use DALR

```python
import torch
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Model/DALR")
model = AutoModel.from_pretrained("Model/DALR")

texts = [
    "There's a kid on a skateboard.",
    "A kid is skateboarding.",
    "A kid is inside the house.",
]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

with torch.no_grad():
    embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output

cosine_sim_0_1 = 1 - cosine(embeddings[0], embeddings[1])
cosine_sim_0_2 = 1 - cosine(embeddings[0], embeddings[2])

print("Cosine similarity between \"%s\" and \"%s\" is: %.3f" % (texts[0], texts[1], cosine_sim_0_1))
print("Cosine similarity between \"%s\" and \"%s\" is: %.3f" % (texts[0], texts[2], cosine_sim_0_2))
```

### Utility: Mine Similar Sentence Pairs

`SimCSE` now provides `most_similar_pairs(...)` to find the most similar
sentence pairs in a sentence list:

```python
from src.tool import SimCSE

simcse = SimCSE("Model/DALR")
pairs = simcse.most_similar_pairs(
    [
        "A kid is skateboarding.",
        "There is a child on a skateboard.",
        "A cat is sleeping on the sofa.",
    ],
    top_k=2,
)
for sent_a, sent_b, score in pairs:
    print(f"{score:.3f} | {sent_a} <> {sent_b}")
```

### Utility: Save / Load Retrieval Index

You can persist a brute-force retrieval index to disk and reload it later:

```python
from src.tool import SimCSE

simcse = SimCSE("Model/DALR")
sentences = [
    "A kid is skateboarding.",
    "There is a child on a skateboard.",
    "A cat is sleeping on the sofa.",
]
simcse.build_index(sentences, use_faiss=False)
simcse.save_index("dalr_index.npz")

another = SimCSE("Model/DALR")
another.load_index("dalr_index.npz")
print(another.search("A child on a skateboard", top_k=2))
```

---

## Evaluation

Run evaluation on SentEval benchmarks:

```bash
python src/evaluation.py \
    --model_name_or_path Model/DALR \
    --pooler cls_before_pooler \
    --task_set sts \
    --mode test
```

To export machine-readable results for experiment tracking, pass `--output_json`:

```bash
python src/evaluation.py \
    --model_name_or_path Model/DALR \
    --pooler cls_before_pooler \
    --task_set sts \
    --mode test \
    --output_json eval_results.json
```

Additional evaluation scripts are provided in `scripts/`:

```bash
bash scripts/run_eval.sh        # STS evaluation
bash scripts/run_eval_coco.sh   # COCO retrieval evaluation
# or choose a specific checkpoint seed:
SEED=1 bash scripts/run_eval_coco.sh
```

---

## Train Your Own Models

### Wiki + Flickr30k

```bash
bash scripts/run_wiki_flickr.sh
```

### Wiki + MS-COCO

```bash
bash scripts/run_wiki_coco.sh
```

You can freely adjust hyperparameters (learning rate, batch size, margins, lambda, etc.) in the respective shell scripts. Key arguments:

| Argument | Description | Default |
|---|---|---|
| `--framework` | Training framework (`simcse` / `mse`) | `mse` |
| `--learning_rate` | Learning rate | `2e-5` |
| `--per_device_train_batch_size` | Batch size per device | `128` |
| `--num_train_epochs` | Number of training epochs | `4` |
| `--lbd` | Weight for distillation loss | `0.01` |
| `--margin1` / `--margin2` | Ranking margins | `0.2` |
| `--distillation_loss` | Distillation loss type | `listmle` |
| `--alpha_` / `--beta_` / `--gamma_` | Loss weights | `0.33 / 1.0 / 1.0` |

---

## FAQ

### Which CUDA / PyTorch versions are supported?

DALR was developed and tested with **PyTorch 1.8.1 + CUDA 11.1**. Newer
PyTorch versions are likely to work but are not officially tested —
please open an Issue if you hit a regression.

### What GPU was used in the paper?

All experiments in the paper were run on a single **NVIDIA Tesla A100
(80 GB)**. Lowering `--per_device_train_batch_size` and increasing
gradient accumulation should let you reproduce training on smaller GPUs.

### Can I run training or evaluation on CPU?

Evaluation works on CPU: the `evaluate()` helper in `src/utils.py`
infers the device from the model's parameters and falls back to CPU
when CUDA is unavailable. Training, however, is GPU-only in practice —
the model relies on CLIP plus a BERT/RoBERTa encoder and ranking
distillation, which is prohibitively slow on CPU.

### What's the difference between `--framework simcse` and `--framework mse`?

- `simcse`: standard SimCSE contrastive objective only.
- `mse` (default in our scripts): adds the DALR ranking distillation
  loss on top of the contrastive objective — this is the setting
  reported in the paper.

### Which pretrained models do I need?

- **CLIP ViT-B/32** — used as the *image teacher* in the paper.
- **SimCSE** and **DiffCSE** — used as the two *text teachers*.
- **BERT-base-uncased** (or **RoBERTa-base**) — the student language
  model. Place them under `Model/` as shown in
  [Download Datasets](#download-datasets).

### Which hyperparameters most influence final performance?

In our experiments the most sensitive knobs are:

- `--lbd` — the weight of the ranking distillation loss
- `--margin1` / `--margin2` — ranking margins (paper uses `0.2`)
- `--learning_rate` — typical range `1e-5` to `2e-5`

The defaults in `scripts/run_wiki_flickr.sh` and `scripts/run_wiki_coco.sh`
correspond to the configurations reported in the paper.

### How is the best checkpoint selected during training?

The development set of STS-B is evaluated every 125 training steps and
the best-performing checkpoint is retained.

---

## Troubleshooting

### `ImportError: cannot import name 'X' from 'transformers'`

You're likely on a newer `transformers` release than what DALR was
developed against. Pin the version from `requirements.txt`:

```bash
pip install "transformers==4.8.2"
```

### `torch.cuda.OutOfMemoryError` during training

Reduce the per-device batch size and increase gradient accumulation so
the effective batch size stays the same. Edit
`scripts/run_wiki_flickr.sh` / `scripts/run_wiki_coco.sh`:

```bash
BATCH=64            # or 32
# and pass --gradient_accumulation_steps 2 (or 4) to the python call
```

### `FileNotFoundError: Model/clip/ViT-B-32.pt`

The CLIP checkpoint was not downloaded into the expected location.
Either let CLIP fetch it on first use (it will be cached under
`~/.cache/clip/`) or download manually and place it at
`Model/clip/ViT-B-32.pt` exactly as shown under
[Download Datasets](#download-datasets).

### `KeyError: 'STSBenchmark'` when running evaluation

The SentEval downstream datasets are not in place. From the repo root:

```bash
cd SentEval/data/downstream/
bash download_dataset.sh
```

### Training stalls at 0% or the very first step

Almost always one of:

- The image paths in the feature JSON (`--feature_file`) don't resolve
  under `--image_root` — check both flags against your local layout.
- The Wiki1M text file is missing or has zero lines.
- CLIP failed to load and silently fell back; check the early logs.

### Loss diverges or STS-B Spearman never improves

Two settings to try first:

- Lower the learning rate (`LR=1e-5` instead of `2e-5`).
- Lower the distillation weight (`LBD=0.005`) or the ranking margins.

If the model still does not improve after a few hundred steps, the
text or image teacher likely failed to load — verify by printing a
similarity matrix from each teacher on a few captions.

---

## Project Structure

```
DALR/
├── clip/                   # CLIP model utilities
├── data/                   # Data directory (datasets downloaded here)
├── figure/                 # Figures used in the paper / README
├── scripts/                # Training and evaluation shell scripts
│   ├── run_wiki_flickr.sh
│   ├── run_wiki_coco.sh
│   ├── run_eval.sh
│   └── run_eval_coco.sh
├── SentEval/               # SentEval toolkit (evaluation)
├── src/                    # Core source code
│   ├── model_dalr.py       # DALR model definition
│   ├── train_mix.py        # Main training script
│   ├── data.py             # Dataset and data loading
│   ├── evaluation.py       # SentEval evaluation
│   ├── teachers.py         # Teacher model wrappers
│   ├── utils.py            # Utility functions
│   ├── vit.py              # Vision Transformer implementation
│   ├── xbert.py            # Extended BERT utilities
│   ├── tool.py             # Miscellaneous tools
│   └── randaugment.py      # RandAugment data augmentation
├── requirements.txt
├── LICENSE
├── CONTRIBUTING.md
├── README.md
└── README_zh.md
```

---

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@inproceedings{he-etal-2025-dalr,
    title = "{DALR}: Dual-level Alignment Learning for Multimodal Sentence Representation Learning",
    author = "He, Kang  and
      Ding, Yuzhe  and
      Wang, Haining  and
      Li, Fei  and
      Teng, Chong  and
      Ji, Donghong",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2025",
    year = "2025",
    pages = "3586--3601",   
}
```

---

## Acknowledgements

- Evaluation is powered by the [SentEval toolkit](https://github.com/facebookresearch/SentEval); we adopt the modified version from [SimCSE](https://github.com/princeton-nlp/SimCSE).
- Part of our code is adapted from [MCSE](https://github.com/uds-lsv/MCSE) and [KDMCSE](https://github.com/duyngtr16061999/KDMCSE).

---

## Contributing

We welcome contributions! Please read our [Contributing Guide](CONTRIBUTING.md) to get started. Feel free to open an [Issue](https://github.com/kangverse/DALR/issues) or submit a [Pull Request](https://github.com/kangverse/DALR/pulls).
