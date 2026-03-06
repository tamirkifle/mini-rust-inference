#!/usr/bin/env bash
# benchmark_comparison.sh — commit 17.3
#
# Side-by-side throughput comparison: this engine vs llama.cpp.
#
# USAGE
#   ./scripts/benchmark_comparison.sh [options]
#
# OPTIONS (also settable via environment variables)
#   --model   PATH    Path to a GGUF model file used by llama.cpp    (LLAMA_MODEL)
#   --llama   PATH    Path to the llama.cpp benchmark binary         (LLAMA_CPP_BIN)
#                       accepts: llama-bench, llama-cli, or main
#   --prompt  N       Number of prompt tokens                        (N_PROMPT, default: 64)
#   --gen     N       Number of tokens to generate                   (N_GEN,    default: 128)
#   --threads N       Thread count for both engines                  (N_THREADS, default: all cores)
#   --warmup  N       Warmup iterations before timing                (N_WARMUP, default: 2)
#   --help            Print this message and exit
#
# EXAMPLES
#   # Our engine only (no GGUF required):
#   ./scripts/benchmark_comparison.sh
#
#   # Full comparison with llama.cpp:
#   LLAMA_MODEL=~/models/llama-7b.Q4_0.gguf \
#   LLAMA_CPP_BIN=~/llama.cpp/build/bin/llama-bench \
#   ./scripts/benchmark_comparison.sh --prompt 64 --gen 128 --threads 8
#
# OUTPUT
#   Prints a formatted comparison table:
#   ┌──────────────────────────┬────────────┬──────────────┬─────────────┐
#   │ Engine                   │ Prompt tok │ Gen tok      │   tok/s     │
#   ├──────────────────────────┼────────────┼──────────────┼─────────────┤
#   │ llm-inference-engine     │         64 │          128 │      XXX.X  │
#   │ llama.cpp (llama-bench)  │         64 │          128 │      YYY.Y  │
#   └──────────────────────────┴────────────┴──────────────┴─────────────┘
#
# NOTES
#   - Our engine is benchmarked via `cargo bench --bench compare_llamacpp`.
#     The bench uses a zero-weight proxy model (no GGUF) to measure raw kernel
#     throughput. For a real model comparison, set LLAMA_MODEL and point
#     LLAMA_CPP_BIN at a llama.cpp build.
#   - llama.cpp timing is extracted from llama-bench's markdown table output
#     (--output json is not used to avoid version-specific requirements).
#   - Both engines are given N_WARMUP warm-up iterations before timing.

set -euo pipefail

# ── defaults ───────────────────────────────────────────────────────────────────
: "${N_PROMPT:=64}"
: "${N_GEN:=128}"
: "${N_WARMUP:=2}"
: "${LLAMA_MODEL:=}"
: "${LLAMA_CPP_BIN:=}"

# Auto-detect core count
if command -v nproc &>/dev/null; then
  : "${N_THREADS:=$(nproc)}"
elif command -v sysctl &>/dev/null; then
  : "${N_THREADS:=$(sysctl -n hw.logicalcpu 2>/dev/null || echo 4)}"
else
  : "${N_THREADS:=4}"
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CARGO="${CARGO_BIN:-$HOME/.cargo/bin/cargo}"
RESULT_FILE="$REPO_ROOT/target/bench_compare_result.txt"


# ── argument parsing ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)   LLAMA_MODEL="$2";   shift 2 ;;
    --llama)   LLAMA_CPP_BIN="$2"; shift 2 ;;
    --prompt)  N_PROMPT="$2";      shift 2 ;;
    --gen)     N_GEN="$2";         shift 2 ;;
    --threads) N_THREADS="$2";     shift 2 ;;
    --warmup)  N_WARMUP="$2";      shift 2 ;;
    --help|-h)
      sed -n '3,50p' "${BASH_SOURCE[0]}" | grep '^#' | sed 's/^# \?//'
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

# ── helpers ────────────────────────────────────────────────────────────────────
hr() { printf '%.0s─' {1..68}; echo; }
pad_right() { printf "%-${2}s" "$1"; }
pad_left()  { printf "%${2}s"  "$1"; }

# ── banner ─────────────────────────────────────────────────────────────────────
echo
echo "  LLM Inference Engine — Benchmark Comparison"
echo "  $(date '+%Y-%m-%d %H:%M:%S')  |  prompt=$N_PROMPT  gen=$N_GEN  threads=$N_THREADS"
hr

# ── step 1: build release benchmarks ─────────────────────────────────────────
echo
echo "  [1/3] Building release benchmarks …"
cd "$REPO_ROOT"
"$CARGO" build --benches --release -q 2>&1 || {
  echo "  ERROR: cargo build --benches failed" >&2; exit 1
}
echo "  Done."

# ── step 2: run our engine bench ──────────────────────────────────────────────
echo
echo "  [2/3] Running our engine bench  (warmup=$N_WARMUP, N_GEN=$N_GEN) …"
echo "        (This takes ~30 s; Criterion runs multiple statistical samples)"

# Export settings so compare_llamacpp.rs can read them at bench time
export BENCH_N_PROMPT="$N_PROMPT"
export BENCH_N_GEN="$N_GEN"
export BENCH_N_WARMUP="$N_WARMUP"

# Run the bench, capturing combined output; Criterion writes to stderr
BENCH_OUTPUT=$("$CARGO" bench --bench compare_llamacpp --release 2>&1 \
  -- --output-format criterion 2>&1 || true)

# Extract mean throughput from Criterion's thrpt line:
#   "comparison/our_engine/n128  thrpt:  [59.123 elem/s 60.456 elem/s 61.789 elem/s]"
# We want the middle (mean) value — the 5th whitespace-separated field after "thrpt:"
OUR_TOKS_PER_SEC=$(echo "$BENCH_OUTPUT" \
  | grep -E 'thrpt:' \
  | tail -1 \
  | grep -oE '\[.*\]' \
  | tr -d '[]' \
  | awk '{print $3}' \
  || echo "")

OUR_UNIT=$(echo "$BENCH_OUTPUT" \
  | grep -E 'thrpt:' \
  | tail -1 \
  | grep -oE '\[.*\]' \
  | tr -d '[]' \
  | awk '{print $4}' \
  || echo "elem/s")

# Normalise to tok/s (elem/s == tok/s when Throughput::Elements(N_GEN) is used)
if [[ -z "$OUR_TOKS_PER_SEC" ]]; then
  OUR_TOKS_PER_SEC="N/A"
fi

echo "  Our engine:  $OUR_TOKS_PER_SEC $OUR_UNIT"
echo "engine_tok_s=$OUR_TOKS_PER_SEC" > "$RESULT_FILE"


# ── step 3: run llama.cpp bench (optional) ─────────────────────────────────────
LLAMA_TOKS_PER_SEC="N/A"
LLAMA_LABEL="llama.cpp (not configured)"

if [[ -n "$LLAMA_CPP_BIN" && -x "$LLAMA_CPP_BIN" ]]; then
  if [[ -z "$LLAMA_MODEL" || ! -f "$LLAMA_MODEL" ]]; then
    echo
    echo "  [3/3] WARNING: LLAMA_CPP_BIN is set but LLAMA_MODEL is missing or not a file."
    echo "        Set LLAMA_MODEL=<path/to/model.gguf> to run the llama.cpp comparison."
    LLAMA_LABEL="llama.cpp (model not found)"
  else
    LLAMA_BIN_NAME=$(basename "$LLAMA_CPP_BIN")
    LLAMA_LABEL="llama.cpp ($LLAMA_BIN_NAME)"
    echo
    echo "  [3/3] Running llama.cpp ($LLAMA_BIN_NAME) …"
    echo "        model:  $LLAMA_MODEL"
    echo "        prompt: $N_PROMPT tokens  gen: $N_GEN tokens  threads: $N_THREADS"

    # llama-bench supports: --model, --n-prompt, --n-gen, -t, --warmup
    # llama-cli / main supports different flags — we try llama-bench first
    if echo "$LLAMA_BIN_NAME" | grep -q "bench"; then
      LLAMA_OUTPUT=$("$LLAMA_CPP_BIN" \
        --model   "$LLAMA_MODEL" \
        --n-prompt "$N_PROMPT" \
        --n-gen   "$N_GEN" \
        -t        "$N_THREADS" \
        --warmup  "$N_WARMUP" \
        2>&1 || true)

      # llama-bench markdown table format:
      # | model | size | ... | t/s prompt | t/s gen |
      # Look for the eval (generation) t/s value
      LLAMA_TOKS_PER_SEC=$(echo "$LLAMA_OUTPUT" \
        | grep -v '^|.*---' \
        | grep '|' \
        | tail -1 \
        | awk -F'|' '{
            for(i=1;i<=NF;i++) {
              gsub(/[[:space:]]/,"",$i);
              if($i ~ /^[0-9]+(\.[0-9]+)?(±[0-9.]+)?$/) last=$i
            }
            print last
          }' \
        || echo "")
    else
      # llama-cli / main: use --n-predict for gen tokens
      LLAMA_OUTPUT=$("$LLAMA_CPP_BIN" \
        --model     "$LLAMA_MODEL" \
        --n-predict "$N_GEN" \
        --threads   "$N_THREADS" \
        --prompt    "$(printf 'word %.0s' $(seq 1 $N_PROMPT))" \
        2>&1 || true)

      # Parse "llama_print_timings:        eval time = ... tokens per second"
      LLAMA_TOKS_PER_SEC=$(echo "$LLAMA_OUTPUT" \
        | grep 'eval time' \
        | grep -oE '[0-9]+\.[0-9]+ tokens per second' \
        | grep -oE '^[0-9]+\.[0-9]+' \
        || echo "")
    fi

    if [[ -z "$LLAMA_TOKS_PER_SEC" ]]; then
      LLAMA_TOKS_PER_SEC="parse-error"
      echo "  WARNING: could not parse tok/s from llama.cpp output."
      echo "  Raw output (last 10 lines):"
      echo "$LLAMA_OUTPUT" | tail -10 | sed 's/^/    /'
    else
      echo "  llama.cpp:   $LLAMA_TOKS_PER_SEC tok/s"
    fi

    echo "llama_tok_s=$LLAMA_TOKS_PER_SEC" >> "$RESULT_FILE"
  fi
else
  echo
  echo "  [3/3] llama.cpp comparison skipped."
  echo "        Set LLAMA_CPP_BIN=<path> and LLAMA_MODEL=<path> to enable."
fi

# ── results table ──────────────────────────────────────────────────────────────
echo
hr
printf "  %-34s  %6s  %6s  %12s\n" "Engine" "Prompt" "Gen" "tok/s"
hr
printf "  %-34s  %6s  %6s  %12s\n" \
  "llm-inference-engine (proxy model)" "$N_PROMPT" "$N_GEN" "$OUR_TOKS_PER_SEC"
printf "  %-34s  %6s  %6s  %12s\n" \
  "$LLAMA_LABEL" "$N_PROMPT" "$N_GEN" "$LLAMA_TOKS_PER_SEC"
hr
echo
echo "  Results written to: $RESULT_FILE"
echo "  Note: 'proxy model' = zero-weight in-memory fixture (pure kernel throughput)."
echo "  For a real model comparison, set LLAMA_MODEL and LLAMA_CPP_BIN."
echo

