"""Training-loop helpers used by `train_mix.py`.

This module bridges the project to the SentEval toolkit and exposes two
small utilities that the main training loop relies on:

* `evaluate` — run an STS-Benchmark evaluation and return a few scalars.
* `inf_train_gen` — wrap a DataLoader so the training loop can `next()` it
  forever without worrying about epoch boundaries.
"""

import sys
from typing import Any, Dict, Iterator, Optional, Union

import torch

# Set path to SentEval
PATH_TO_SENTEVAL = './SentEval'
PATH_TO_DATA = './SentEval/data'

sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval


def evaluate(
    model: torch.nn.Module,
    tokenizer: Any,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, float]:
    """Run STS-Benchmark evaluation against the given model.

    Args:
        model: A transformer model exposing the HuggingFace forward interface.
        tokenizer: A HuggingFace tokenizer compatible with `model`.
        device: Optional device override. When `None`, the device is inferred
            from `model.parameters()` and falls back to CPU if no CUDA device
            is available.

    Returns:
        Dict with keys `eval_stsb_spearman`, `eval_stsb_align`, and
        `eval_stsb_uniform`.
    """
    # Infer the model's device when not given, falling back to CPU so the
    # script also runs on machines without CUDA.
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def prepare(params, samples):
        return

    def batcher(params, batch):
        sentences = [' '.join(s) for s in batch]
        batch = tokenizer.batch_encode_plus(
            sentences,
            return_tensors='pt',
            padding=True,
        )
        for k in batch:
            batch[k] = batch[k].to(device)
        with torch.no_grad():
            outputs = model(**batch, output_hidden_states=True, return_dict=True)
        return outputs.last_hidden_state[:, 0].cpu()  # unpooled [CLS] output in BERT

    # Set params for SentEval (fastmode)
    params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
    params['classifier'] = {
        'nhid': 0,
        'optim': 'rmsprop',
        'batch_size': 128,
        'tenacity': 3,
        'epoch_size': 2,
    }

    se = senteval.engine.SE(params, batcher, prepare)
    tasks = ['STSBenchmark']
    results = se.eval(tasks)
    stsb_spearman = results['STSBenchmark']['dev']['spearman'][0]
    stsb_align = results['STSBenchmark']['dev']['align_loss']
    stsb_uniform = results['STSBenchmark']['dev']['uniform_loss']

    return {
        "eval_stsb_spearman": stsb_spearman,
        "eval_stsb_align": stsb_align,
        "eval_stsb_uniform": stsb_uniform,
    }


def inf_train_gen(trainloader: torch.utils.data.DataLoader) -> Iterator[Any]:
    """Yield batches from `trainloader` forever, restarting at each epoch end."""
    while True:
        yield from trainloader
