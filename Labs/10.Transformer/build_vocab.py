"""
build_vocab.py
--------------
Scan one or more dataset files (TSV: f<TAB>F) produced by
make_integral_dataset.py and emit a vocab.json file compatible with TinyLM.

The JSON layout mirrors the Taylor-series vocab:

    {
        "token2id": {"<PAD>": 0, "<SOS>": 1, ...},
        "id2token": {"0": "<PAD>", "1": "<SOS>", ...},
        "PAD": 0, "SOS": 1, "EOS": 2, "SEP": 3
    }

Special tokens are always assigned the first four IDs (0–3) so they are
independent of the data tokens and can be hard-coded in the model config.

Usage
-----
    # Single dataset file
    python build_vocab.py integrals.txt --out vocab_integrals.json

    # Merge two datasets (e.g. integrals + Taylor series)
    python build_vocab.py integrals.txt taylor.txt --out vocab_merged.json

    # Preview without writing
    python build_vocab.py integrals.txt --dry_run
"""

import argparse
import json
import sys
from pathlib import Path

# ------------------------------------------------------------------
# Special tokens — fixed IDs that TinyLM's Config keys must match
# ------------------------------------------------------------------
SPECIAL_TOKENS = ["<PAD>", "<SOS>", "<EOS>", "<SEP>"]
PAD_ID, SOS_ID, EOS_ID, SEP_ID = 0, 1, 2, 3


def collect_tokens(paths: list[str]) -> list[str]:
    """Return a sorted list of unique data tokens across all files."""
    seen: set[str] = set()
    for path in paths:
        text = Path(path).read_text(encoding="utf-8")
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                continue                   # skip malformed lines
            for part in parts:
                for tok in part.split():   # tokens already space-separated
                    seen.add(tok)
    return sorted(seen)                    # deterministic order


def build_vocab(data_tokens: list[str]) -> dict:
    """
    Combine special tokens + data tokens into id↔token mappings.
    Data tokens are sorted alphabetically so the vocab is stable
    across runs on the same data.
    """
    all_tokens = SPECIAL_TOKENS + data_tokens
    token2id = {tok: idx for idx, tok in enumerate(all_tokens)}
    id2token  = {str(idx): tok for idx, tok in enumerate(all_tokens)}

    return {
        "token2id": token2id,
        "id2token":  id2token,
        # Convenience: special-token IDs as top-level keys (mirrors Taylor vocab)
        "PAD": PAD_ID,
        "SOS": SOS_ID,
        "EOS": EOS_ID,
        "SEP": SEP_ID,
        "VOCAB_SIZE": len(all_tokens),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build TinyLM-compatible vocab from integral dataset files")
    parser.add_argument("files", nargs="+",
                        help="TSV dataset files produced by make_integral_dataset.py")
    parser.add_argument("--out", default="vocab.json",
                        help="Output JSON file (default: vocab.json)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print vocab info but do not write the file")
    args = parser.parse_args()

    # Check all files exist
    for f in args.files:
        if not Path(f).exists():
            print(f"ERROR: file not found: {f}", file=sys.stderr)
            sys.exit(1)

    data_tokens = collect_tokens(args.files)
    vocab = build_vocab(data_tokens)

    print(f"Data tokens : {len(data_tokens)}")
    print(f"Total vocab : {vocab['VOCAB_SIZE']}  "
          f"(4 special + {len(data_tokens)} data)")
    print(f"Special IDs : PAD={PAD_ID}, SOS={SOS_ID}, "
          f"EOS={EOS_ID}, SEP={SEP_ID}")
    print()
    print("Data tokens:")
    print(" ", " | ".join(data_tokens))

    if not args.dry_run:
        Path(args.out).write_text(
            json.dumps(vocab, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        print(f"\nWrote {args.out}")
    else:
        print("\n(dry run — nothing written)")


if __name__ == "__main__":
    main()
