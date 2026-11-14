"""Score n-gram drafting policies offline, from the outputs of a plain (no-drafter) run.

An n-gram drafter's guesses depend only on the text so far, and under greedy decoding the
target's output is fixed, so acceptance can be computed exactly without a GPU.

  python ngram_sim.py plain_edit.json --tokenizer ricursiveAI/Ripple-65B-Preview

tokens/step is an upper bound on speedup: it assumes drafting and wider verification are free.
"""
import argparse, json, re


class NGram:
    """Match the last n tokens earlier in the context and propose the k tokens that followed.
    ns: match lengths to try, longest first. sticky: keep following the source after a mismatch."""

    def __init__(self, ns, k, sticky=False):
        self.ns, self.k, self.sticky = sorted(ns, reverse=True), k, sticky
        self.idx, self.done, self.ptr = {n: {} for n in self.ns}, 0, None

    def draft(self, ctx):
        L = len(ctx)
        for e in range(self.done, L):  # index every n-gram that ends before the current position
            for n in self.ns:
                if e >= n:
                    self.idx[n][tuple(ctx[e - n:e])] = e
        self.done = L
        if self.sticky and self.ptr is not None and self.ptr < L:
            return ctx[self.ptr:self.ptr + self.k]
        for n in self.ns:
            j = self.idx[n].get(tuple(ctx[L - n:L])) if L >= n else None
            if j is not None:
                self.ptr = j
                return ctx[j:j + self.k]
        self.ptr = None
        return []

    def feedback(self, accepted, advanced):
        # after a mismatch, assume one token was substituted and stay aligned with the source
        keep = self.sticky and self.ptr is not None and accepted > 0
        self.ptr = self.ptr + advanced if keep else None


POLICIES = {
    "n=3  k=10  (classic prompt lookup)": lambda: NGram([3], 10),
    "n=24 k=64  (llama.cpp ngram-mod default)": lambda: NGram([24], 64),
    "n=12 k=64": lambda: NGram([12], 64),
    "n=8  k=32": lambda: NGram([8], 32),
    "longest 4..24 k=32  (suffix-match style)": lambda: NGram(range(4, 25), 32),
    "longest 4..24 k=32 + re-sync after edits": lambda: NGram(range(4, 25), 32, sticky=True),
}


def simulate(prompt, out, pol):
    ctx, i, steps, acc, drafted = list(prompt), 0, 0, 0, 0
    while i < len(out):
        d = pol.draft(ctx)
        a = 0
        while a < len(d) and i + a < len(out) and d[a] == out[i + a]:
            a += 1
        adv = min(a + 1, len(out) - i)  # accepted draft tokens plus one token from the target
        pol.feedback(a, adv)
        ctx += out[i:i + adv]
        i, steps, acc, drafted = i + adv, steps + 1, acc + a, drafted + len(d)
    return steps, acc, drafted


ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+", help="JSON files written by bench.py for a run without any drafter")
ap.add_argument("--tokenizer", default="words", help="Hugging Face repo id, or 'words' for a rough built-in split")
a = ap.parse_args()
if a.tokenizer == "words":
    enc = lambda s: re.findall(r"\w+|\s+|[^\w\s]", s)
else:
    from tokenizers import Tokenizer
    tok = Tokenizer.from_pretrained(a.tokenizer)
    enc = lambda s: tok.encode(s, add_special_tokens=False).ids

for f in a.runs:
    rows = [(enc(r["prompt"]), enc(r["text"])) for r in json.load(open(f))]
    n = sum(len(o) for _, o in rows)
    print(f"\n{f}: {len(rows)} outputs, {n} tokens")
    print(f"{'policy':44s} {'tokens/step':>11s} {'from drafts':>12s} {'draft hit rate':>15s}")
    for name, make in POLICIES.items():
        s = [simulate(p, o, make()) for p, o in rows]
        steps, acc, drafted = (sum(x[j] for x in s) for j in range(3))
        print(f"{name:44s} {n / steps:11.2f} {acc / n:12.1%} {acc / max(drafted, 1):15.1%}")
