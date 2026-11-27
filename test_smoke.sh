#!/usr/bin/env bash
# Smoke test: verifies an OpenAI-compatible endpoint is alive and answers a
# minimal chat completion.
#
# Usage: ./test_smoke.sh [base_url] [model]
#   base_url defaults to http://localhost:8000/v1
#   model defaults to the first model in /v1/models
#   auth: set OPENAI_API_KEY (or API_KEY) if the endpoint requires it
#
# Exit codes: 0 = pass, 1 = endpoint unreachable, 2 = bad response shape,
#             3 = empty/degenerate completion
set -u

BASE_URL="${1:-http://localhost:8000/v1}"
MODEL="${2:-}"
TIMEOUT=120
API_KEY="${OPENAI_API_KEY:-${API_KEY:-}}"
AUTH=()
if [ -n "$API_KEY" ]; then
    AUTH=(-H "Authorization: Bearer $API_KEY")
fi

fail() { echo "FAIL: $*" >&2; exit "${2:-1}"; }

# --- 1. list models -----------------------------------------------------------
MODELS_JSON=$(curl -s -m 10 "${AUTH[@]}" "$BASE_URL/models") || fail "cannot reach $BASE_URL/models" 1
if [ -z "$MODEL" ]; then
    MODEL=$(echo "$MODELS_JSON" | python3 -c \
        'import sys,json;print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null) \
        || fail "no models in /models response: $MODELS_JSON" 2
fi
echo "endpoint: $BASE_URL"
echo "model:    $MODEL"

# --- 2. minimal chat completion ----------------------------------------------
RESP=$(curl -s -m "$TIMEOUT" "${AUTH[@]}" "$BASE_URL/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"$MODEL\",
        \"messages\": [{\"role\": \"user\", \"content\": \"Reply with exactly the word: PONG\"}],
        \"max_tokens\": 32768,
        \"temperature\": 0
    }") || fail "chat completion request timed out/failed" 1

CONTENT=$(echo "$RESP" | python3 -c \
    'import sys,json;print(json.load(sys.stdin)["choices"][0]["message"]["content"])' 2>/dev/null) \
    || fail "bad response shape: ${RESP:0:500}" 2

# some serving stacks place reasoning in a separate field; content may be empty
if [ -z "$CONTENT" ]; then
    CONTENT=$(echo "$RESP" | python3 -c \
        'import sys,json;print(json.load(sys.stdin)["choices"][0]["message"].get("reasoning_content",""))' 2>/dev/null)
fi

echo "reply:    ${CONTENT:0:200}"
if echo "$CONTENT" | grep -qi "PONG"; then
    echo "PASS"
    exit 0
fi
fail "expected PONG in reply, got: ${CONTENT:0:200}" 3
