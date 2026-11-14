# riafterdark

A copy fast path for SGLang's DFlash speculative decoding, to be measured on
Ripple-65B Preview with its DFlash2 drafter on code-editing prompts.

When the last N generated tokens also appear earlier in the request's context, the
tokens that followed that earlier occurrence replace the head of the drafter's block.
The target still verifies every token.

Status: the patch is written and unit-tested on CPU. It has not run on a GPU. See
`results.md`.

## Layout

- `patch/dflash_copy_sglang_0.5.19.diff`: the diff against sglang v0.5.19.
- `patch/dflash_copy.py`: copy of the new file in the diff, so the test runs without sglang.
- `tests/test_copy_matcher.py`: CPU test of the matcher against `ngram_sim.py`'s policy.
- `bench/`: `make_edit_prompts.py`, `bench.py`, `ngram_sim.py`.
- `modal/download_weights.py`: pulls both checkpoints to the Modal volume `ripple-weights`.
- `modal/sandbox.py`: builds sglang v0.5.19 from source and starts a GPU sandbox.
- `modal/run_bench.sh`: applies the patch, serves with the official recipe, runs the bench.

## Flags

All off by default.

| Variable | Default | Meaning |
| --- | --- | --- |
| `SGLANG_DFLASH_COPY` | 0 | Enable the copy fast path (greedy requests only). |
| `SGLANG_DFLASH_COPY_N` | 8 | Match length. A comma list tries the longest first. |
| `SGLANG_DFLASH_COPY_MAX` | 0 | Max copied tokens per step. 0 means block size minus 1. |
| `SGLANG_DFLASH_COPY_STICKY` | 0 | Keep following the source after a mismatch. |
| `SGLANG_DFLASH_COPY_DEBUG` | 0 | Assert alignment with `seq_lens` and log copy stats. |

## Run

```bash
python tests/test_copy_matcher.py          # CPU, needs torch

modal run modal/download_weights.py
python modal/sandbox.py B200:2             # prints the sandbox id
# copy this repo to /repo in the sandbox, then in the sandbox:
/repo/modal/run_bench.sh baseline
python /repo/bench/ngram_sim.py /repo/runs/baseline_edit.json --tokenizer ricursiveAI/Ripple-65B-Preview
/repo/modal/run_bench.sh patched 8
python /repo/bench/bench.py compare /repo/runs/baseline_edit.json /repo/runs/patched_edit.json
```
