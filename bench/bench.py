"""Time a vLLM/SGLang OpenAI-compatible endpoint, then compare two runs.

  python bench.py run --name plain  --model ricursiveAI/Ripple-65B-Preview
  python bench.py run --name dflash --model ricursiveAI/Ripple-65B-Preview
  python bench.py compare plain.json dflash.json
"""
import argparse, json, statistics, time
from openai import OpenAI

PROMPTS = [
    "Write a Python function that merges overlapping intervals. Include tests.",
    "Implement an LRU cache in Python without using functools.",
    "Write a bash script that finds the 10 largest files under a directory.",
    "Implement Dijkstra's algorithm in Python with a binary heap.",
    "Write a Python class for a thread-safe rate limiter using a token bucket.",
    "Write a SQL query for the top 3 customers by revenue per region, and explain it.",
    "Implement a trie with insert, search and prefix-count in Python.",
    "Write a Python script that tails a log file and prints lines matching a regex.",
    "Refactor a recursive Fibonacci into an iterative one and explain the complexity.",
    "Write a Python function to parse an ISO 8601 duration string into seconds.",
]


def run(a):
    client = OpenAI(base_url=a.url, api_key="x")
    prompts = [json.loads(l)["prompt"] for l in open(a.prompts)] if a.prompts else PROMPTS
    rows = []
    for i, p in enumerate(prompts[: a.n] * a.repeats):
        if i == 0:  # warm-up, not recorded
            client.chat.completions.create(model=a.model, messages=[{"role": "user", "content": p}], max_tokens=32)
        t0, t_first, text, usage, chunks = time.perf_counter(), None, "", None, 0
        for ch in client.chat.completions.create(
            model=a.model, messages=[{"role": "user", "content": p}], max_tokens=a.max_tokens,
            temperature=a.temperature, top_p=a.top_p, seed=0, stream=True, stream_options={"include_usage": True},
        ):
            if ch.usage:
                usage = ch.usage
            if ch.choices:
                d = ch.choices[0].delta
                piece = (getattr(d, "reasoning_content", None) or "") + (d.content or "")
                if piece and t_first is None:
                    t_first = time.perf_counter()
                text += piece
                chunks += bool(piece)
        t1 = time.perf_counter()
        n = usage.completion_tokens if usage else chunks  # fall back if the server sends no usage
        rows.append({"prompt": p, "tokens": n, "ttft_s": t_first - t0, "total_s": t1 - t0,
                     "decode_tps": (n - 1) / (t1 - t_first), "text": text})
        print(f"[{i + 1}] {n} tok  {rows[-1]['decode_tps']:.1f} tok/s  ttft {rows[-1]['ttft_s']:.2f}s")
    json.dump(rows, open(f"{a.name}.json", "w"), indent=1)
    print(f"median decode: {statistics.median(r['decode_tps'] for r in rows):.1f} tok/s -> {a.name}.json")


def compare(a):
    base, fast = (json.load(open(f)) for f in (a.base, a.fast))
    assert [r["prompt"] for r in base] == [r["prompt"] for r in fast], "prompt sets differ"
    med = lambda rows, k: statistics.median(r[k] for r in rows)
    ratios = [f["decode_tps"] / b["decode_tps"] for b, f in zip(base, fast)]
    same = sum(b["text"] == f["text"] for b, f in zip(base, fast))
    print(f"baseline   {med(base, 'decode_tps'):7.1f} tok/s   ttft {med(base, 'ttft_s'):.2f}s")
    print(f"sped up    {med(fast, 'decode_tps'):7.1f} tok/s   ttft {med(fast, 'ttft_s'):.2f}s")
    print(f"speedup    {statistics.median(ratios):.2f}x median   (min {min(ratios):.2f}x, max {max(ratios):.2f}x)")
    print(f"end-to-end {med(base, 'total_s') / med(fast, 'total_s'):.2f}x")
    print(f"identical outputs: {same}/{len(base)}")


ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd", required=True)
r = sub.add_parser("run")
r.add_argument("--name", required=True)
r.add_argument("--model", required=True)
r.add_argument("--url", default="http://localhost:8000/v1")
r.add_argument("--prompts", help="JSONL with a 'prompt' field; default is 10 built-in coding prompts")
r.add_argument("--n", type=int, default=10)
r.add_argument("--repeats", type=int, default=2)
r.add_argument("--max-tokens", type=int, default=512)
r.add_argument("--temperature", type=float, default=0.0, help="0 = greedy; outputs only match across runs at 0")
r.add_argument("--top-p", type=float, default=1.0)
r.set_defaults(fn=run)
c = sub.add_parser("compare")
c.add_argument("base")
c.add_argument("fast")
c.set_defaults(fn=compare)
args = ap.parse_args()
args.fn(args)
