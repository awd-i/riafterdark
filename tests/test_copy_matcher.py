"""CPU check: DFlashCopyDrafter driven like the worker reproduces ngram_sim's NGram policy exactly."""
import importlib.util, random, sys, torch
spec = importlib.util.spec_from_file_location("dflash_copy", (sys.argv[1] if len(sys.argv) > 1 else str(__import__("pathlib").Path(__file__).resolve().parents[1] / "patch" / "dflash_copy.py"))); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

class NGram:  # from ngram_sim.py
    def __init__(self, ns, k, sticky=False):
        self.ns, self.k, self.sticky = sorted(ns, reverse=True), k, sticky
        self.idx, self.done, self.ptr = {n: {} for n in self.ns}, 0, None
    def draft(self, ctx):
        L = len(ctx)
        for e in range(self.done, L):
            for n in self.ns:
                if e >= n: self.idx[n][tuple(ctx[e - n:e])] = e
        self.done = L
        if self.sticky and self.ptr is not None and self.ptr < L: return ctx[self.ptr:self.ptr + self.k]
        for n in self.ns:
            j = self.idx[n].get(tuple(ctx[L - n:L])) if L >= n else None
            if j is not None:
                self.ptr = j; return ctx[j:j + self.k]
        self.ptr = None; return []
    def feedback(self, accepted, advanced):
        keep = self.sticky and self.ptr is not None and accepted > 0
        self.ptr = self.ptr + advanced if keep else None

class Req:
    def __init__(s, rid, p): s.rid, s.p = rid, p
    def get_fill_ids(s): return s.p

def run(ns, sticky, seed, block=16):
    r = random.Random(seed)
    base = [r.randrange(50) for _ in range(300)]
    prompt = base[:200]
    out = []
    while len(out) < 600:  # output = noisy copies of prompt spans
        a = r.randrange(150); seg = prompt[a:a + r.randrange(5, 60)]
        out += [t if r.random() > 0.03 else r.randrange(50) for t in seg]
    d = m.DFlashCopyDrafter(",".join(map(str, ns)), block - 1, sticky, True)
    ref = NGram(ns, block - 1, sticky)
    req = Req("a", prompt)
    # worker view: bonus = out[i]; committed prefix = prompt + out[:i]
    i, ctx = 0, list(prompt) + [out[0]]
    while i + 1 < len(out):
        got = d.propose([req], torch.tensor([out[i]]), torch.tensor([len(prompt) + i]))[0]
        want = ref.draft(ctx)
        assert got == want, (i, got, want)
        cand = [out[i]] + got + [-1] * (block - 1 - len(got))   # rest: drafter always wrong
        a = 0
        while a < block - 1 and i + 1 + a < len(out) and cand[1 + a] == out[i + 1 + a]: a += 1
        if i + 1 + a >= len(out): break
        committed = out[i + 1:i + 1 + a + 1]          # accepted drafts + new bonus
        row = committed + [0] * (block - len(committed))
        d.record(["a"], torch.tensor([row]), torch.tensor([a + 1], dtype=torch.int32))
        ref.feedback(a, a + 1); ctx += committed; i += a + 1
    return d.stats
for ns, st in [([8], False), ([3], False), (list(range(4, 25)), False), (list(range(4, 25)), True)]:
    print(ns[:3], st, [run(ns, st, s) for s in range(3)])
print("ok")
