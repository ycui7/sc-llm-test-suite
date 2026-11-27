#!/bin/bash
# GLM-5.3-EXL3 full qualification battery — all traffic via Bifrost, zero interference.
set -u
cd "$(dirname "$0")"
KEY="$(cat "$HOME/.config/qc/apikey")"   # house rule: keys live in chmod-600 files, never in scripts
M="gx10/glm-5.3-exl3"
BASE="${QC_BASE_URL:?gateway /v1 URL is not committed — set QC_BASE_URL}"
R="$(pwd)/results/glm53-exl3-$(date +%m%d_%H%M)"
mkdir -p "$R"
echo "RESULTS_DIR=$R"

echo "==== PHASE1 TOOL-ROUNDTRIP ===="
python3 qc_4b_tool_file.py --base-url "$BASE" --model "$M" --api-key "$KEY" >"$R/tool.txt" 2>&1
echo "tool exit=$?"; tail -6 "$R/tool.txt"

echo "==== PHASE1 STREAM ===="
python3 qc_stream.py --base-url "$BASE" --model "$M" --api-key "$KEY" >"$R/stream.txt" 2>&1
echo "stream exit=$?"; tail -6 "$R/stream.txt"

echo "==== PHASE2 BENCHY (solo) ===="
( for i in $(seq 1 90); do
    ssh -o BatchMode=yes -o ConnectTimeout=5 gx10-f9e3 'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; grep MemAvailable /proc/meminfo' >>"$R/gpu-head.log" 2>/dev/null
    ssh -o BatchMode=yes -o ConnectTimeout=5 gx10-0696 'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; grep MemAvailable /proc/meminfo' >>"$R/gpu-worker.log" 2>/dev/null
    sleep 10
  done ) &
SAMPLER=$!
llama-benchy --base-url "$BASE" --api-key "$KEY" --model "$M" \
  --pp 2048 --tg 512 --concurrency 1 2 4 --format md >"$R/benchy.md" 2>&1
echo "benchy exit=$?"; grep -E "pp2048|tg512|concurrency|tok/s|Throughput" "$R/benchy.md" | tail -20
kill $SAMPLER 2>/dev/null

echo "==== PHASE3 NEEDLES 2048@262144 (solo) ===="
python3 test_needles.py --base-url "$BASE" --api-key "$KEY" --model "$M" --ctx 262144 --seed 1234 >"$R/needles.log" 2>&1
echo "needles exit=$?"; tail -12 "$R/needles.log"

echo "==== PHASE4 PREFIX-CACHE TTFT ===="
python3 - "$BASE" "$KEY" "$M" >"$R/ttft.txt" 2>&1 <<'PYEOF'
import sys, json, time, urllib.request
base, key, model = sys.argv[1], sys.argv[2], sys.argv[3]
prompt = ("Explain the water cycle in detail. " * 1200)  # ~8-10K tokens, repeated
def ttft():
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 40, "temperature": 0.0, "stream": True}
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "***" + key})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            if line.startswith(b"data:") and b"choices" in line:
                return time.time() - t0
    return -1.0
for i in range(3):
    print(f"run{i+1} ttft={ttft():.2f}s", flush=True)
PYEOF
cat "$R/ttft.txt"
echo "==== BATTERY DONE ===="
