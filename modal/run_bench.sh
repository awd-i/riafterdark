#!/usr/bin/env bash
# Runs inside the Modal sandbox (modal/sandbox.py). NOT YET RUN on a GPU.
#   run_bench.sh baseline            flag off
#   run_bench.sh patched [N]         SGLANG_DFLASH_COPY=1, N = match length(s), default 8
# Expects this repo at /repo, weights at /w, sglang v0.5.19 source at /opt/sglang.
set -euo pipefail
MODE=$1; N=${2:-8}
T=/w/Ripple-65B-Preview; D=/w/Ripple-65B-Preview-DFlash2
cd /opt/sglang
git apply --check /repo/patch/dflash_copy_sglang_0.5.19.diff 2>/dev/null && git apply /repo/patch/dflash_copy_sglang_0.5.19.diff
if [ "$MODE" = patched ]; then export SGLANG_DFLASH_COPY=1 SGLANG_DFLASH_COPY_N=$N SGLANG_DFLASH_COPY_DEBUG=1; fi
cd $T/inference/server
python serve.py --model-path $T --speculative-draft-model-path $D > /tmp/server_$MODE.log 2>&1 &
SERVER=$!
trap 'kill $SERVER' EXIT
until curl -sf http://127.0.0.1:30000/health > /dev/null; do kill -0 $SERVER; sleep 5; done
mkdir -p /repo/runs && cd /repo/runs
[ -f edit_prompts.jsonl ] || python /repo/bench/make_edit_prompts.py
B="python /repo/bench/bench.py run --model ripple --url http://127.0.0.1:30000/v1"
$B --name ${MODE}_scratch
$B --name ${MODE}_edit --prompts edit_prompts.jsonl
