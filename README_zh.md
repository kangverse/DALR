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

### 论文链接

- arXiv： https://arxiv.org/abs/2506.21096
- ACL Anthology（Findings ACL 2025）： https://aclanthology.org/2025.findings-acl.183/

---

## 目录

- [快速开始](#快速开始)
  - [环境配置](#环境配置)
  - [下载数据集](#下载数据集)
- [使用 DALR](#使用-dalr)
- [评估](#评估)
- [训练自己的模型](#训练自己的模型)
- [常见问题](#常见问题)
- [故障排查](#故障排查)
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
│       └── ViT-B-32.pt
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

预训练模型（SimCSE、DiffCSE、BERT-base、CLIP ViT-B/32）可从 [Hugging Face](https://huggingface.co/) 下载，放入 `Model/` 目录。

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
| `--margin1` / `--margin2` | 排序间距 | `0.2` |
| `--distillation_loss` | 蒸馏损失类型 | `listmle` |
| `--alpha_` / `--beta_` / `--gamma_` | 损失权重 | `0.33 / 1.0 / 1.0` |

---

## 常见问题

### 支持哪些 CUDA / PyTorch 版本？

DALR 在 **PyTorch 1.8.1 + CUDA 11.1** 下开发和测试。更高版本的 PyTorch
理论上应该可用，但未经官方测试——如果遇到回归问题请提一个 Issue。

### 论文使用什么 GPU？

所有论文实验都在单张 **NVIDIA Tesla A100 (80 GB)** 上完成。降低
`--per_device_train_batch_size` 并增加梯度累积步数，可在更小的 GPU 上
复现训练。

### 能在 CPU 上训练或评估吗？

评估支持 CPU：`src/utils.py` 中的 `evaluate()` 会从模型参数自动推断设备，
在没有 CUDA 时回退到 CPU。但训练实际上是 GPU-only 的——模型依赖 CLIP
加上 BERT/RoBERTa 编码器和排序蒸馏，在 CPU 上跑会极慢。

### `--framework simcse` 和 `--framework mse` 有什么区别？

- `simcse`：仅使用标准 SimCSE 对比目标。
- `mse`（脚本默认）：在对比目标之上叠加 DALR 排序蒸馏损失，这是论文
  中报告的配置。

### 需要哪些预训练模型？

- **CLIP ViT-B/32** —— 论文中作为*图像教师*。
- **SimCSE** 和 **DiffCSE** —— 作为两个*文本教师*。
- **BERT-base-uncased**（或 **RoBERTa-base**）—— 学生语言模型。

按 [下载数据集](#下载数据集) 中所示放入 `Model/` 目录。

### 哪些超参对最终性能影响最大？

我们实验中最敏感的几个：

- `--lbd` —— 排序蒸馏损失的权重
- `--margin1` / `--margin2` —— 排序间距（论文用 `0.2`）
- `--learning_rate` —— 典型范围 `1e-5` 到 `2e-5`

`scripts/run_wiki_flickr.sh` 和 `scripts/run_wiki_coco.sh` 中的默认值
对应论文报告的配置。

### 训练过程中如何选择最佳 checkpoint？

每 125 个训练步在 STS-B 开发集上评估一次，保留最佳 checkpoint。

---

## 故障排查

### `ImportError: cannot import name 'X' from 'transformers'`

你装的 `transformers` 版本比 DALR 开发时新。按 `requirements.txt` 固定版本：

```bash
pip install "transformers==4.8.2"
```

### 训练时 `torch.cuda.OutOfMemoryError`

降低每设备批大小并提高梯度累积步数，使有效 batch size 不变。修改
`scripts/run_wiki_flickr.sh` / `scripts/run_wiki_coco.sh`：

```bash
BATCH=64            # 或 32
# 并向 python 调用传 --gradient_accumulation_steps 2（或 4）
```

### `FileNotFoundError: Model/clip/ViT-B-32.pt`

CLIP checkpoint 未下载到指定位置。要么让 CLIP 在首次使用时自动下载
（会缓存到 `~/.cache/clip/`），要么手动下载并放在 `Model/clip/ViT-B-32.pt`，
完全按照 [下载数据集](#下载数据集) 的示意。

### 评估时报 `KeyError: 'STSBenchmark'`

SentEval 的下游数据集没下。在仓库根目录执行：

```bash
cd SentEval/data/downstream/
bash download_dataset.sh
```

### 训练卡在 0% 或第一步

通常是以下情况之一：

- 特征 JSON（`--feature_file`）里的图片路径在 `--image_root` 下找不到
  —— 检查两个参数是否对应你本地的数据布局。
- Wiki1M 文本文件缺失或者是空的。
- CLIP 加载失败但被静默忽略 —— 检查训练初期的日志。

### Loss 发散，或 STS-B Spearman 一直不涨

先尝试两个调整：

- 降低学习率（`LR=1e-5` 替代 `2e-5`）。
- 降低蒸馏权重（`LBD=0.005`）或排序间距。

跑了几百步还是没改善，多半是文本或图像教师没加载成功 —— 可以打印
教师在几条 caption 上的相似度矩阵来验证。

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
