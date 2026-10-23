"""
make_integral_dataset.py
------------------------
Generate a dataset of (f, F) integral pairs in the same format as the
TinyLM Taylor-series dataset:

    <SOS> f(x) <SEP> F(x) <EOS>

Strategy (Lample & Charton forward generation):
    1. Build a random antiderivative F(x) from a tree of operators.
    2. Differentiate: f = dF/dx  (sp.diff only — always fast).
    3. Tokenise with SymPy's default str() printer.
    4. Filter: length cap, no special functions, no infinities.
    5. Drop any additive constant from F (unless --keep_constant): the
       integral of f is defined only up to a constant, and nothing in f
       says which constant to use, so a model cannot learn it.

Attempts run in worker processes (ProcessPoolExecutor), in batches; each
attempt is interrupted after --timeout s, so a slow SymPy call cannot stall
the run. Pairs are written as they are found.

Usage
-----
    python make_integral_dataset.py \
        --n_samples 50000 \
        --max_ops   6     \
        --out_file  integrals.txt \
        --seed      42    \
        --workers   4

Output: one tab-separated line per sample  f_tokens<TAB>F_tokens

Dependencies: sympy
"""

import argparse
import random
import re
import sys
from typing import Optional

import sympy as sp
from sympy.abc import x

# ---------------------------------------------------------------------------
# Non-elementary / special-function symbols — skip these
# ---------------------------------------------------------------------------
SKIP_FUNCS = (
    sp.erf, sp.erfi, sp.erfc,
    sp.Ei, sp.Si, sp.Ci, sp.Shi, sp.Chi,
    sp.li, sp.Li,
    sp.fresnels, sp.fresnelc,
    sp.hyper, sp.meijerg,
    sp.uppergamma, sp.lowergamma, sp.gamma,
    sp.polygamma, sp.digamma, sp.trigamma,
    sp.elliptic_e, sp.elliptic_f, sp.elliptic_k, sp.elliptic_pi,
    sp.besselj, sp.bessely, sp.besseli, sp.besselk,
    sp.Piecewise,
)

# ---------------------------------------------------------------------------
# Leaf nodes
# ---------------------------------------------------------------------------
INT_CONSTS  = [sp.Integer(n) for n in [-3, -2, -1, 2, 3, 4, 5]]
RATIONALS   = [sp.Rational(1, 2), sp.Rational(1, 3), sp.Rational(2, 3),
               sp.Rational(1, 4), sp.Rational(3, 2)]
MATH_CONSTS = [sp.pi, sp.E]
LEAVES = [x] * 6 + INT_CONSTS + RATIONALS + MATH_CONSTS

# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------
UNARY_OPS = [
    sp.sin, sp.cos, sp.tan,
    sp.asin, sp.acos, sp.atan,
    sp.sinh, sp.cosh, sp.tanh,
    sp.exp, sp.log, sp.sqrt,
]

def _add(a, b): return a + b
def _sub(a, b): return a - b
def _mul(a, b): return a * b
def _div(a, b): return a / b
def _pow(a, b): return a ** b

BINARY_OPS     = [_add, _sub, _mul, _div, _pow]
BINARY_WEIGHTS = [3, 3, 3, 1, 1]


# ---------------------------------------------------------------------------
# Random expression tree
# ---------------------------------------------------------------------------

def random_expr(n_ops: int, rng: random.Random):
    if n_ops == 0:
        return rng.choice(LEAVES)

    n_binary = rng.randint(max(0, n_ops - 3), n_ops)
    n_unary  = n_ops - n_binary

    ops = (rng.choices(BINARY_OPS, weights=BINARY_WEIGHTS, k=n_binary)
           + rng.choices(UNARY_OPS, k=n_unary))
    rng.shuffle(ops)

    nodes = [rng.choice(LEAVES) for _ in range(n_binary + 1)]

    try:
        for op in ops:
            if op in UNARY_OPS:
                idx = rng.randrange(len(nodes))
                nodes[idx] = op(nodes[idx])
            else:
                if len(nodes) < 2:
                    nodes.append(rng.choice(LEAVES))
                i, j = rng.sample(range(len(nodes)), 2)
                a, b = nodes[i], nodes[j]
                result = op(a, b)
                for idx in sorted([i, j], reverse=True):
                    nodes.pop(idx)
                nodes.append(result)
    except Exception:
        return None

    return nodes[0] if nodes else None


# ---------------------------------------------------------------------------
# The constant of integration
# ---------------------------------------------------------------------------

def drop_additive_constant(F):
    """Remove the terms of a sum that do not contain x:  g(x) + c  ->  g(x).

    An antiderivative is defined only up to an additive constant. The
    random expression trees often produce one (e.g. F = x**2 + 3/2), but
    f = dF/dx carries no trace of it, so asking a model to predict it is
    asking for the impossible. Only *explicit* top-level constants are
    removed; a constant hidden inside a product, e.g. E*(cosh(x) + 1), is
    kept, since removing it would mean rewriting the expression.
    """
    if isinstance(F, sp.Add):
        kept = [term for term in F.args if term.has(x)]
        if kept and len(kept) < len(F.args):
            return sp.Add(*kept)
    return F


# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------

def expr_to_tokens(expr) -> Optional[str]:
    try:
        s = str(expr)
    except Exception:
        return None
    s = s.replace('exp(1)', 'E')
    tokens = re.findall(r'[A-Za-z_]\w*|\d+\.\d+|\d+|[+\-*/\*\*\(\),]|/|\*\*|\*', s)
    # Reject large integer literals (powers of trig constants, etc.)
    if any(tok.isdigit() and int(tok) > 9999 for tok in tokens):
        return None
    if len(tokens) > 80:
        return None
    return ' '.join(tokens)


# ---------------------------------------------------------------------------
# Worker — one attempt per call; runs in a subprocess
# ---------------------------------------------------------------------------

def _worker(args_tuple):
    """One generation attempt. Called in a subprocess via ProcessPoolExecutor."""
    max_ops, seed, keep_constant = args_tuple
    rng = random.Random(seed)

    n_ops = rng.choices(
        range(1, max_ops + 1),
        weights=[max_ops - k + 1 for k in range(1, max_ops + 1)]
    )[0]

    F_raw = random_expr(n_ops, rng)
    if F_raw is None:
        return None

    try:
        F_sym = sp.sympify(F_raw)
    except Exception:
        return None

    if not keep_constant:
        F_sym = drop_additive_constant(F_sym)

    if not F_sym.has(x):
        return None
    if F_sym.has(sp.zoo, sp.oo, sp.nan, sp.I):
        return None

    try:
        f_sym = sp.diff(F_sym, x)
    except Exception:
        return None

    if f_sym == 0 or not f_sym.has(x):
        return None
    if f_sym.has(sp.zoo, sp.oo, sp.nan, sp.I):
        return None

    for expr in (f_sym, F_sym):
        if expr.has(*SKIP_FUNCS):
            return None
        if expr.has(sp.RootSum, sp.RootOf):
            return None
        if expr.has(sp.AccumBounds):        # SymPy's "somewhere in [a, b]": not a function
            return None

    f_tok = expr_to_tokens(f_sym)
    F_tok = expr_to_tokens(F_sym)

    if f_tok is None or F_tok is None:
        return None

    return f_tok, F_tok


# ---------------------------------------------------------------------------
# Dataset generator
# ---------------------------------------------------------------------------

def _worker_batch(args_tuple):
    """Several generation attempts in one task (fewer round trips to the
    main process). Each attempt gets at most `timeout` seconds: SymPy can
    occasionally run for a very long time, and SIGALRM interrupts it."""
    import signal
    max_ops, seeds, keep_constant, timeout = args_tuple

    def _too_slow(signum, frame):
        raise TimeoutError

    signal.signal(signal.SIGALRM, _too_slow)
    results = []
    for s in seeds:
        signal.alarm(timeout)
        try:
            results.append(_worker((max_ops, s, keep_constant)))
        except Exception:                    # TimeoutError or a SymPy error
            results.append(None)
        finally:
            signal.alarm(0)
    return results


def generate_dataset(n_samples: int, max_ops: int, out_file: str,
                     seed: int = 42,
                     max_attempts_factor: int = 20,
                     timeout_secs: int = 3,
                     n_workers: int = 4,
                     keep_constant: bool = False,
                     progress: bool = True,
                     chunk: int = 50) -> None:
    """
    Generate up to n_samples pairs with distinct f and write them to out_file.

    Attempt k uses the random seed  seed + k, so a run is reproducible up to
    the order in which the worker processes finish. Each task sent to a worker
    holds `chunk` attempts, and a new task is sent as soon as any finishes, so
    the workers are never idle. Pairs are written as soon as they are found,
    so a run that is stopped early (e.g. by a batch-job time limit) keeps
    what it has made.
    """
    import time
    from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

    seen = set()
    n_pairs = 0
    attempts = 0
    max_attempts = n_samples * max_attempts_factor
    next_seed = seed
    t0 = last_report = time.time()

    with ProcessPoolExecutor(max_workers=n_workers) as pool, \
         open(out_file, 'w') as fh:

        pending = set()

        def submit():
            nonlocal next_seed, attempts
            seeds = list(range(next_seed, next_seed + chunk))
            next_seed += chunk
            attempts += chunk
            pending.add(pool.submit(_worker_batch,
                                    (max_ops, seeds, keep_constant, timeout_secs)))

        for _ in range(2 * n_workers):                 # keep every worker busy
            submit()

        while pending and n_pairs < n_samples:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                pending.remove(fut)
                for result in fut.result():
                    if result is None:
                        continue
                    f_tok, F_tok = result
                    if f_tok in seen or n_pairs >= n_samples:
                        continue
                    seen.add(f_tok)
                    fh.write(f'{f_tok}\t{F_tok}\n')
                    n_pairs += 1
                if n_pairs < n_samples and attempts < max_attempts:
                    submit()
            if progress and time.time() - last_report > 10:
                last_report = time.time()
                fh.flush()
                print(f'  {n_pairs:,} pairs from {attempts:,} attempts '
                      f'({time.time() - t0:.0f} s)', file=sys.stderr, flush=True)

        for fut in pending:                            # not needed any more
            fut.cancel()

    print(f'\nGenerated {n_pairs} pairs in {attempts} attempts '
          f'({n_pairs/max(attempts,1):.1%} yield rate, {time.time() - t0:.0f} s)',
          file=sys.stderr)
    print(f'Wrote {n_pairs} samples to {out_file}', file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Generate integral dataset for TinyLM')
    parser.add_argument('--n_samples',  type=int, default=50_000)
    parser.add_argument('--max_ops',    type=int, default=6)
    parser.add_argument('--out_file',   type=str, default='integrals.txt')
    parser.add_argument('--seed',       type=int, default=42)
    parser.add_argument('--timeout',    type=int, default=3,
                        help='Per-sample subprocess timeout in seconds')
    parser.add_argument('--workers',    type=int, default=4,
                        help='Parallel worker processes')
    parser.add_argument('--no_progress', action='store_true',
                        help='no progress messages')
    parser.add_argument('--keep_constant', action='store_true',
                        help='keep additive constants in F (the old behaviour)')
    args = parser.parse_args()

    generate_dataset(
        n_samples=args.n_samples,
        max_ops=args.max_ops,
        out_file=args.out_file,
        seed=args.seed,
        timeout_secs=args.timeout,
        n_workers=args.workers,
        keep_constant=args.keep_constant,
        progress=not args.no_progress,
    )


if __name__ == '__main__':
    main()
