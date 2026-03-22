# DALR

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.7+](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://www.python.org/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

[English](README.md) | [中文](README_zh.md)

## 概述

我们提出了 **DALR**（**D**ual-level **A**lignment **L**earning for multimodal sentence **R**epresentation Learning），一种面向多模态句子表示学习的双层对齐方法。

为了实现跨模态细粒度对齐，我们提出了一种跨模态对齐方法来缓解 *跨模态错位偏差*（CMB）问题。为了缓解 *模态内语义偏离*（ISD）问题，我们将排序蒸馏与全局对齐学习相结合，从而有效地对齐模态内表示。

下图展示了模型的整体架构：

![DALR 模型架构](figure/model.png)

---

## 目录

- [快速开始](#快速开始)
  - [环境配置](#环境配置)
  - [下载数据集](#下载数据集)
- [使用 DALR](#使用-dalr)
- [评估](#评估)
- [训练自己的模型](#训练自己的模型)
- [项目结构](#项目结构)
- [引用](#引用)
- [致谢](#致谢)
- [参与贡献](#参与贡献)

---

## 快速开始

### 环境配置

建议先创建虚拟环境：

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

安装 PyTorch（CUDA 11.1）：

```bash
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 \
    -f https://download.pytorch.org/whl/torch_stable.html
```

如果 CUDA 版本 < 11 或仅使用 CPU：

```bash
pip install torch==1.8.1
```

安装其余依赖：

```bash
pip install -r requirements.txt
```

### 下载数据集

从官方网站下载 Flickr30k 和 MS-COCO 数据集，并按以下结构组织：

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
│       └── ViT-L-14.pt
```

**Wiki1M**（用于文本训练）：

```bash
wget https://huggingface.co/datasets/princeton-nlp/datasets-for-simcse/resolve/main/wiki1m_for_simcse.txt \
    -P data/
```

**SentEval 评估数据集**（来自 [SimCSE](https://github.com/princeton-nlp/SimCSE)）：

```bash
cd SentEval/data/downstream/
bash download_dataset.sh
```

预训练模型（SimCSE、DiffCSE、BERT-base、CLIP ViT-L/14）可从 [Hugging Face](https://huggingface.co/) 下载，放入 `Model/` 目录。

---

## 使用 DALR

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

---

## 评估

在 SentEval 基准上运行评估：

```bash
python src/evaluation.py \
    --model_name_or_path Model/DALR \
    --pooler cls_before_pooler \
    --task_set sts \
    --mode test
```

`scripts/` 目录下还提供了其他评估脚本：

```bash
bash scripts/run_eval.sh        # STS 评估
bash scripts/run_eval_coco.sh   # COCO 检索评估
```

---

## 训练自己的模型

### Wiki + Flickr30k

```bash
bash scripts/run_wiki_flickr.sh
```

### Wiki + MS-COCO

```bash
bash scripts/run_wiki_coco.sh
```

你可以在对应的 shell 脚本中自由调整超参数（学习率、批大小、margin、lambda 等）。主要参数说明：

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--framework` | 训练框架（`simcse` / `mse`） | `mse` |
| `--learning_rate` | 学习率 | `2e-5` |
| `--per_device_train_batch_size` | 每设备批大小 | `128` |
| `--num_train_epochs` | 训练轮数 | `4` |
| `--lbd` | 蒸馏损失权重 | `0.01` |
| `--margin1` / `--margin2` | 排序间距 | `0.18` |
| `--distillation_loss` | 蒸馏损失类型 | `listmle` |
| `--alpha_` / `--beta_` / `--gamma_` | 损失权重 | `0.33 / 1.0 / 1.0` |

---

## 项目结构

```
DALR/
├── clip/                   # CLIP 模型工具
├── data/                   # 数据目录（数据集下载至此）
├── figure/                 # 论文 / README 中使用的图片
├── scripts/                # 训练和评估 shell 脚本
│   ├── run_wiki_flickr.sh
│   ├── run_wiki_coco.sh
│   ├── run_eval.sh
│   └── run_eval_coco.sh
├── SentEval/               # SentEval 评估工具包
├── src/                    # 核心源代码
│   ├── model_dalr.py       # DALR 模型定义
│   ├── train_mix.py        # 主训练脚本
│   ├── data.py             # 数据集与数据加载
│   ├── evaluation.py       # SentEval 评估
│   ├── teachers.py         # 教师模型封装
│   ├── utils.py            # 工具函数
│   ├── vit.py              # Vision Transformer 实现
│   ├── xbert.py            # 扩展 BERT 工具
│   ├── tool.py             # 其他工具
│   └── randaugment.py      # RandAugment 数据增强
├── requirements.txt
├── LICENSE
├── CONTRIBUTING.md
├── README.md
└── README_zh.md
```

---

## 引用

如果本项目对你的研究有帮助，请考虑引用：

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

## 致谢

- 评估部分使用了 [SentEval 工具包](https://github.com/facebookresearch/SentEval)；我们采用了 [SimCSE](https://github.com/princeton-nlp/SimCSE) 中的修改版本。
- 部分代码参考了 [MCSE](https://github.com/uds-lsv/MCSE) 和 [KDMCSE](https://github.com/duyngtr16061999/KDMCSE)。

---

## 参与贡献

欢迎贡献！请阅读 [贡献指南](CONTRIBUTING.md) 了解详情。你可以随时提交 [Issue](https://github.com/kangverse/DALR/issues) 或发起 [Pull Request](https://github.com/kangverse/DALR/pulls)。
