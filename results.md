# Results

No GPU results yet. Nothing below the "Verified" section has been run.

## Status

The Modal workspace I was given (`azx0616`) has no payment method, so Modal refused
every GPU type. The server has not been started, with or without the patch.

## Verified (CPU only)

- sglang builds from source at v0.5.19 in the Modal image (CUDA 13, torch 2.13.0+cu130).
- Both checkpoints downloaded to a Modal volume (target 130 GB, DFlash2 drafter 8 GB).
- `tests/test_copy_matcher.py`: the matcher, driven the way the worker drives it
  (bonus token in, committed block out), proposes exactly the tokens that
  `ngram_sim.py`'s `NGram` policy proposes, for n=8, n=3, longest 4..24, and
  longest 4..24 with re-sync. Synthetic token streams, 3 seeds each.
- black 26.1.0 and isort 7.0.0 (the versions sglang pins) pass on the three touched files.

## Not verified

- That the patched server starts and answers a request.
- That greedy outputs are identical with the flag on. This is the check that matters.
- That the flag-off path is byte-identical. By reading the diff, flag off adds one
  `None` attribute and three `is not None` checks, but it has not been run.
- Any speedup. No tokens/s numbers exist for baseline or patched.
- The ngram_sim table. It needs baseline outputs from the real model.
- Loop exclusion. No outputs to inspect.
- `modal/run_bench.sh` has never been executed.
- Whether the official recipe (`trtllm_mha` attention, 2x B200) runs on Modal at all.

## Findings from reading 0.5.19

- The spot named in the task exists unchanged in 0.5.19 (`forward_batch_generation`,
  the three `draft_tokens` lines before `# --- 2) Target verify.`).
- Overlap scheduling does make `req.output_ids` lag by one step. `ngram_worker.py`
  says so and splices the previous step's tokens back in. The patch tracks committed
  tokens inside the worker instead, so it does not depend on the scheduler mode.
- The greedy accept path compares `draft_tokens` with the target argmax, so replacing
  draft tokens cannot change greedy output, only how many tokens are accepted per step.
- DFlash2 has a selector with a sampled path that carries draft probabilities. The
  override is skipped unless the whole batch is greedy.

## Prior art

This is not a new method. llama.cpp chains n-gram lookup with DFlash, and Snowflake
and Baseten ship similar hybrids on other engines.
