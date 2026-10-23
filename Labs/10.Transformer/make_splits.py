#!/usr/bin/env python3
"""
make_splits.py  --  set aside a fixed test set, and build the training files
=============================================================================

Comparing models trained on different data is fair only if they are tested on
the same events. This script fixes the test set once and for all:

  1. data/integrals_test.txt   the 2,997 pairs that the notebook has always
                               used as its test set (same seed, same split),
                               with additive constants removed;
  2. data/integrals_small.txt  the other 46,952 pairs of integrals_noconst.txt,
                               for training and validation;

    python make_splits.py                                  # steps 1 and 2
    python make_splits.py --extra data/integrals_new.txt   # and step 3
"""
import argparse
from pathlib import Path

from integral_dataset import split_indices

p = argparse.ArgumentParser(description="Fix the test set; build training files.")
p.add_argument("--source", default="data/integrals_noconst.txt")
p.add_argument("--extra", default=None, help="newly generated pairs to add (large set)")
p.add_argument("--seed", type=int, default=42)
p.add_argument("--frac", type=float, default=0.06, help="the notebook's VAL_FRAC = TEST_FRAC")
args = p.parse_args()

lines = [l for l in Path(args.source).read_text().splitlines() if "\t" in l]
train_idx, val_idx, test_idx = split_indices(len(lines), args.frac, args.frac, args.seed)
test = [lines[i] for i in sorted(test_idx)]
small = [lines[i] for i in sorted(train_idx + val_idx)]
Path("data/integrals_test.txt").write_text("\n".join(test) + "\n")
Path("data/integrals_small.txt").write_text("\n".join(small) + "\n")
print(f"test set: {len(test)} pairs;  small training set: {len(small)} pairs")

if args.extra:
    seen = {l.split("\t")[0] for l in lines}        # every f of the original 50k
    large, n_dup = list(small), 0
    for l in Path(args.extra).read_text().splitlines():
        if "\t" not in l:
            continue
        f = l.split("\t")[0]
        if f in seen:
            n_dup += 1
            continue
        seen.add(f)
        large.append(l)
    Path("data/integrals_large.txt").write_text("\n".join(large) + "\n")
    print(f"large training set: {len(large)} pairs "
          f"({len(large) - len(small)} new; {n_dup} duplicates or test-set f's left out)")
