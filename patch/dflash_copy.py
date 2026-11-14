"""Copy fast path for DFLASH (env-gated, greedy only).

When the last N committed tokens of a request also occur earlier in its context
(prompt + output so far), the tokens that followed the earlier occurrence replace the
head of the drafter's block. The target verifies every token as usual.

Committed tokens are tracked here rather than read from req.output_ids, which lags
the true position by one step under overlap scheduling.
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class _ReqState:
    __slots__ = ("ctx", "idx", "done", "ptr", "alive")

    def __init__(self, ctx: List[int], ns: Tuple[int, ...]):
        self.ctx = ctx
        self.idx: Dict[int, Dict[tuple, int]] = {n: {} for n in ns}
        self.done = 0
        self.ptr: Optional[int] = None
        self.alive = True


class DFlashCopyDrafter:
    def __init__(self, ns: str, max_copy: int, sticky: bool, debug: bool):
        self.ns = tuple(sorted({int(x) for x in str(ns).split(",") if x}, reverse=True))
        self.k = int(max_copy)
        self.sticky = bool(sticky)
        self.debug = bool(debug)
        self.states: Dict[str, _ReqState] = {}
        # (rids, out_tokens[bs, block], commit_lens[bs]) of the previous decode step,
        # still on the GPU; pulled to the host at the start of the next step.
        self.pending = None
        self.stats = [0, 0, 0]  # steps, steps with a copy, copied tokens proposed

    def clear(self) -> None:
        self.states.clear()
        self.pending = None

    def record(self, rids, out_tokens: torch.Tensor, commit_lens: torch.Tensor) -> None:
        self.pending = (rids, out_tokens, commit_lens)

    def propose(
        self, reqs, bonus_tokens: torch.Tensor, seq_lens=None
    ) -> List[List[int]]:
        """Advance every request's context and return the copied tokens per request."""
        bs = len(reqs)
        pending, self.pending = self.pending, None
        prev: Dict[str, List[int]] = {}
        bonus_dev = bonus_tokens.reshape(-1)[:bs].to(torch.int64)
        if pending is not None:
            p_rids, p_out, p_commit = pending
            pb = len(p_rids)
            block = p_out.shape[1]
            flat = torch.cat(
                [bonus_dev, p_commit[:pb].to(torch.int64), p_out[:pb].reshape(-1)]
            ).tolist()
            bonus = flat[:bs]
            for j, rid in enumerate(p_rids):
                base = bs + pb + j * block
                prev[rid] = flat[base : base + flat[bs + j]]
        else:
            bonus = bonus_dev.tolist()
        lens = seq_lens.tolist() if (self.debug and seq_lens is not None) else None

        states: Dict[str, _ReqState] = {}
        out: List[List[int]] = []
        for i, req in enumerate(reqs):
            rid = req.rid
            st = self.states.get(rid)
            if st is None:
                st = _ReqState(list(req.get_fill_ids()) + [bonus[i]], self.ns)
            elif st.alive:
                committed = prev.get(rid)
                if committed is None or committed[-1] != bonus[i]:
                    logger.warning(
                        "DFLASH copy: lost track of %s; disabled for it.", rid
                    )
                    st.alive = False
                else:
                    st.ctx.extend(committed)
                    if self.sticky and st.ptr is not None and len(committed) > 1:
                        st.ptr += len(committed)
                    else:
                        st.ptr = None
            states[rid] = st
            if lens is not None and st.alive and len(st.ctx) - 1 != lens[i]:
                raise RuntimeError(
                    f"DFLASH copy misaligned: ctx={len(st.ctx) - 1} seq_len={lens[i]}"
                )
            toks = self._draft(st) if st.alive else []
            self.stats[0] += 1
            self.stats[1] += bool(toks)
            self.stats[2] += len(toks)
            out.append(toks)
        self.states = states  # drops requests that left the batch
        if self.debug and self.stats[0] % 200 == 0:
            logger.info("DFLASH copy stats steps/copy_steps/copied=%s", self.stats)
        return out

    def _draft(self, st: _ReqState) -> List[int]:
        ctx = st.ctx
        L = len(ctx)
        # Index every n-gram that ends before the current position.
        for n in self.ns:
            idx = st.idx[n]
            for e in range(max(st.done, n), L):
                idx[tuple(ctx[e - n : e])] = e
        st.done = L
        if st.ptr is not None and st.ptr < L:
            return ctx[st.ptr : st.ptr + self.k]
        for n in self.ns:
            if L >= n:
                j = st.idx[n].get(tuple(ctx[L - n :]))
                if j is not None:
                    st.ptr = j
                    return ctx[j : j + self.k]
        st.ptr = None
        return []
