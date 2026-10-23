"""
integral_dataset.py
-------------------
PyTorch Dataset and DataLoader for the integral TSV files produced by
make_integral_dataset.py.

The interface mirrors the Taylor-series DataLoader:

    Each batch is a padded LongTensor of shape (B, T) containing
    sequences of the form:

        <SOS>  f_tok_1  f_tok_2 ... f_tok_k  <SEP>
               F_tok_1  F_tok_2 ... F_tok_n  <EOS>  <PAD> ...

    The model's training loop calls:
        logits    = model(batch)
        loss_mask = model.make_loss_mask(batch)
        loss      = (criterion(logits, targets) * loss_mask).sum() / loss_mask.sum()

    For inference, feed everything up to and including <SEP> as the prompt
    and let generate() autoregressively complete.

Classes
-------
    Vocab               — loads vocab.json, exposes encode/decode helpers
    IntegralDataset     — reads TSV, tokenises, truncates
    collate_fn          — pads a list of 1-D LongTensors → (B, T)
    make_dataloaders    — convenience wrapper: train/val/test splits

Example
-------
    from integral_dataset import Vocab, make_dataloaders

    vocab  = Vocab("data/vocab_integrals.json")
    train_loader, val_loader, test_loader = make_dataloaders(
        "data/integrals_small.txt", vocab,
        val_frac=0.06, max_seq_len=128, batch_size=32, seed=42,
        test_path="data/integrals_test.txt")   # the fixed test set

    for batch in train_loader:       # batch: LongTensor (B, T)
        logits    = model(batch)
        loss_mask = model.make_loss_mask(batch)
        ...
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset


# ──────────────────────────────────────────────────────────────────
# Vocabulary
# ──────────────────────────────────────────────────────────────────

class Vocab:
    """
    Thin wrapper around the vocab.json produced by build_vocab.py.

    Attributes (all ints)
    ---------------------
    PAD, SOS, EOS, SEP   — special-token IDs
    size                 — total vocab size (= VOCAB_SIZE)
    """

    def __init__(self, path: str | Path) -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.token2id: dict[str, int] = data["token2id"]
        self.id2token: dict[int, str] = {int(k): v
                                          for k, v in data["id2token"].items()}
        self.PAD:  int = data["PAD"]
        self.SOS:  int = data["SOS"]
        self.EOS:  int = data["EOS"]
        self.SEP:  int = data["SEP"]
        self.size: int = data["VOCAB_SIZE"]

    # ------------------------------------------------------------------
    def encode(self, token_string: str) -> list[int]:
        """
        Convert a space-separated token string to a list of IDs.
        Unknown tokens are mapped to PAD with a warning (safe for inference,
        indicates a vocab mismatch during training — rebuild vocab).
        """
        ids = []
        for tok in token_string.split():
            if tok in self.token2id:
                ids.append(self.token2id[tok])
            else:
                import warnings
                warnings.warn(f"Unknown token {tok!r} → PAD")
                ids.append(self.PAD)
        return ids

    def decode(self, ids: list[int] | torch.Tensor,
               skip_special: bool = True) -> str:
        """
        Convert a list / 1-D tensor of IDs to a human-readable string.
        Stops at EOS. Special tokens are omitted when skip_special=True.
        """
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        special = {self.PAD, self.SOS, self.EOS, self.SEP}
        out = []
        for idx in ids:
            if idx == self.EOS:
                break
            if skip_special and idx in special:
                continue
            out.append(self.id2token.get(idx, "?"))
        return " ".join(out)

    def __repr__(self) -> str:
        return (f"Vocab(size={self.size}, "
                f"PAD={self.PAD}, SOS={self.SOS}, "
                f"EOS={self.EOS}, SEP={self.SEP})")


# ──────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────

class IntegralDataset(Dataset):
    """
    Each item is a 1-D LongTensor:

        [SOS, *f_ids, SEP, *F_ids, EOS]

    Sequences longer than max_seq_len are silently dropped during
    __init__ so every item in the dataset fits in a single batch row.
    """

    def __init__(
        self,
        path: str | Path,
        vocab: Vocab,
        max_seq_len: int = 128,
        indices: Optional[list[int]] = None,   # subset (for train/val/test split)
    ) -> None:
        self.vocab       = vocab
        self.max_seq_len = max_seq_len

        # Load all pairs, then optionally select a subset
        all_pairs = self._load(path)
        if indices is not None:
            all_pairs = [all_pairs[i] for i in indices]

        # Encode and filter
        self.sequences: list[torch.Tensor] = []
        skipped = 0
        for f_str, F_str in all_pairs:
            seq = self._encode_pair(f_str, F_str)
            if seq is None:
                skipped += 1
                continue
            self.sequences.append(seq)

        if skipped:
            import warnings
            warnings.warn(
                f"{skipped} sequences exceeded max_seq_len={max_seq_len} "
                f"and were dropped. Raise --max_ops or max_seq_len.")

    # ------------------------------------------------------------------
    def _load(self, path: str | Path) -> list[tuple[str, str]]:
        pairs = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            pairs.append((parts[0], parts[1]))
        return pairs

    def _encode_pair(self, f_str: str, F_str: str) -> Optional[torch.Tensor]:
        """Build [SOS, *f, SEP, *F, EOS] — return None if too long."""
        v = self.vocab
        ids = ([v.SOS]
               + v.encode(f_str)
               + [v.SEP]
               + v.encode(F_str)
               + [v.EOS])
        if len(ids) > self.max_seq_len:
            return None
        return torch.tensor(ids, dtype=torch.long)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.sequences[idx]

    def prompt(self, idx: int) -> torch.Tensor:
        """
        Return the prompt portion [SOS, *f_ids, SEP] for sample idx.
        Pass this to generate() to autoregressively produce F.
        """
        seq = self.sequences[idx]
        sep_pos = (seq == self.vocab.SEP).nonzero(as_tuple=True)[0]
        if len(sep_pos) == 0:
            return seq          # malformed — return full sequence
        end = int(sep_pos[0].item()) + 1   # include SEP
        return seq[:end]


# ──────────────────────────────────────────────────────────────────
# Collate (pad a list of variable-length tensors)
# ──────────────────────────────────────────────────────────────────

def collate_fn(batch: list[torch.Tensor], pad_id: int) -> torch.Tensor:
    """
    Pad a list of 1-D LongTensors to the same length (right-pad with PAD).
    Returns a (B, T) LongTensor.
    """
    max_len = max(seq.size(0) for seq in batch)
    padded  = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    for i, seq in enumerate(batch):
        padded[i, :seq.size(0)] = seq
    return padded


# ──────────────────────────────────────────────────────────────────
# Convenience: train / val / test split
# ──────────────────────────────────────────────────────────────────

def split_indices(n_total: int, val_frac: float, test_frac: float,
                  seed: int = 42) -> tuple[list[int], list[int], list[int]]:
    """
    Shuffle the line numbers 0..n_total-1 with a fixed seed and cut them
    into train / validation / test lists. Same arguments -> same split.
    """
    rng = random.Random(seed)
    indices = list(range(n_total))
    rng.shuffle(indices)

    n_test = math.ceil(n_total * test_frac)
    n_val  = math.ceil(n_total * val_frac)
    n_train = n_total - n_val - n_test

    return (indices[:n_train],
            indices[n_train : n_train + n_val],
            indices[n_train + n_val :])


def count_pairs(path: str | Path) -> int:
    """Number of f<TAB>F lines in a dataset file."""
    return sum(1 for line in Path(path).read_text().splitlines() if "\t" in line)


def make_dataloaders(
    path: str | Path,
    vocab: Vocab,
    *,
    val_frac:    float = 0.05,
    test_frac:   float = 0.05,
    max_seq_len: int   = 128,
    batch_size:  int   = 64,
    eval_batch_size: Optional[int] = None,
    test_path:   Optional[str | Path] = None,
    num_workers: int   = 0,
    seed:        int   = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load the dataset, split it into train / val / test, return three
    DataLoaders.

    The split is done on the raw line indices (before encoding) so the
    proportions refer to the raw file, not the post-filter dataset.

    test_path       : if given, the test set is read from this file (a fixed
                      test set, so that models trained on different data can
                      be compared fairly), and `path` is split into train and
                      validation only (test_frac is ignored).
    eval_batch_size : batch size of the validation and test loaders
                      (default: batch_size).

    Returns
    -------
    train_loader, val_loader, test_loader
    """
    n_total = count_pairs(path)
    if test_path is not None:
        test_frac = 0.0
    train_idx, val_idx, test_idx = split_indices(n_total, val_frac, test_frac, seed)
    eval_batch_size = eval_batch_size or batch_size

    def _make(file, idx_list, shuffle: bool, bs: int) -> DataLoader:
        ds = IntegralDataset(file, vocab,
                             max_seq_len=max_seq_len,
                             indices=idx_list)
        from functools import partial
        return DataLoader(
            ds,
            batch_size=bs,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=partial(collate_fn, pad_id=vocab.PAD),
        )

    train_loader = _make(path, train_idx, True,  batch_size)
    val_loader   = _make(path, val_idx,   False, eval_batch_size)
    if test_path is None:
        test_loader = _make(path, test_idx, False, eval_batch_size)
    else:
        test_loader = _make(test_path, None, False, eval_batch_size)

    return train_loader, val_loader, test_loader


# ──────────────────────────────────────────────────────────────────
# Checking an answer: is F an antiderivative of f?
# ──────────────────────────────────────────────────────────────────

def is_antiderivative(f_tokens: str, F_tokens: str, timeout: int = 10) -> bool:
    """
    Return True if dF/dx = f, i.e. if F is *an* antiderivative of f.

    An exact string comparison with the dataset's answer is too strict:
    antiderivatives are defined only up to an additive constant, and the
    same function can be written in many ways (x + x**2 or x**2 + x).
    Differentiating the prediction and comparing with f accepts every
    correct answer.

    f_tokens, F_tokens : space-separated token strings, as in the dataset
                         ('**' appears as '* *').
    timeout            : seconds allowed for SymPy (some expressions make
                         simplify() very slow); a timeout counts as False.
                         Not enforced on Windows, which lacks SIGALRM.
    """
    import signal
    import sympy as sp

    x = sp.Symbol('x')

    def parse(tokens):
        return sp.sympify(tokens.replace(' ', ''), locals={'x': x, 'E': sp.E})

    def _timeout(signum, frame):
        raise TimeoutError

    use_alarm = hasattr(signal, 'SIGALRM')
    if use_alarm:
        old = signal.signal(signal.SIGALRM, _timeout)
        signal.alarm(timeout)
    try:
        difference = sp.diff(parse(F_tokens), x) - parse(f_tokens)
        if sp.simplify(difference) == 0:
            return True
        # simplify() can miss identities; test numerically at a few points
        values = [complex(difference.subs(x, v).evalf()) for v in (0.37, 1.21, 2.53)]
        return all(abs(v) < 1e-8 for v in values)
    except Exception:                  # unparsable prediction, timeout, ...
        return False
    finally:
        if use_alarm:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)
