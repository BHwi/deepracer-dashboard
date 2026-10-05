#!/usr/bin/env bash
# 가짜 DRfC: 실제 bin/activate.sh 의 '-e <실험>' 동작과 dr-* 명령의 인터페이스만 흉내 낸다.
_MOCK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export DR_DIR="$_MOCK_DIR"
_exp=""
OPTIND=1
while getopts ":e:" _o; do case $_o in e) _exp="$OPTARG" ;; esac; done
OPTIND=1
set -a
source "$DR_DIR/system.env"
if [[ -n "$_exp" ]]; then
  if [[ ! -d "$DR_DIR/experiments/$_exp" ]]; then echo "Experiment directory $DR_DIR/experiments/$_exp does not exist."; return 1; fi
  export DR_EXPERIMENT_NAME="$_exp"
  source "$DR_DIR/experiments/$_exp/run.env"
fi
set +a
_S="$DR_DIR/.mock_state"; mkdir -p "$_S"
_BUCKET="$DR_DIR/data/minio/${DR_LOCAL_S3_BUCKET:-bucket}"

dr-update-env() { return 0; }

dr-upload-custom-files() {
  local src="$DR_DIR/custom_files"
  [[ -n "$DR_EXPERIMENT_NAME" ]] && src="$DR_DIR/experiments/$DR_EXPERIMENT_NAME/custom_files"
  echo "Uploading files to s3://${DR_LOCAL_S3_BUCKET}/custom_files/"
  mkdir -p "$_BUCKET/custom_files" && cp -r "$src"/. "$_BUCKET/custom_files/"
}

_mock_stack_blocked() {   # 실제 DRfC 는 'docker stack ps <스택> | wc -l' 이 1 보다 크면(종료된 작업의 기록 포함) 시작을 거부한다
  grep -qx "$1" "$_S/stack_alive" "$_S/stack_stale" 2>/dev/null
}

dr-start-training() {
  local OPTIND opt wipe=""
  if _mock_stack_blocked "deepracer-${DR_RUN_ID:-0}"; then echo "ERROR: Processes running in stack deepracer-${DR_RUN_ID:-0}. Stop training with dr-stop-training."; return 1; fi
  while getopts ":whqsavr:" opt; do case $opt in w) wipe=1 ;; esac; done
  if [[ -f "$_S/pid" ]] && kill -0 "$(cat "$_S/pid")" 2>/dev/null; then
    echo "ERROR: Processes running in stack deepracer-0. Stop training with dr-stop-training."; return 1; fi
  if [[ ! -f "$_BUCKET/custom_files/reward_function.py" ]]; then
    echo "Training aborted. Configuration files were not found."; echo "You might have to run dr-upload-custom files."; return 1; fi
  local target="$_BUCKET/$DR_LOCAL_S3_MODEL_PREFIX"
  if [[ -n "$(ls -A "$target" 2>/dev/null)" ]]; then
    if [[ -z "$wipe" ]]; then echo "Selected path s3://$DR_LOCAL_S3_BUCKET/$DR_LOCAL_S3_MODEL_PREFIX exists. Delete it, or use -w option. Exiting."; return 1
    else echo "Wiping path s3://$DR_LOCAL_S3_BUCKET/$DR_LOCAL_S3_MODEL_PREFIX."; rm -rf "$target"; fi
  fi
  mkdir -p "$target/model" "$target/metrics"
  echo "Using image ${DR_SIMAPP_SOURCE}:${DR_SIMAPP_VERSION}"
  nohup python3 "$DR_DIR/tools/mock_trainer.py" train >/dev/null 2>&1 &
  echo $! > "$_S/pid"; disown
  echo "Starting training (mock)"
}

dr-stop-training() {
  if [[ -n "$MOCK_STOP_FAIL" ]]; then echo "Error: could not stop (mock)"; return 1; fi
  if [[ -f "$_S/pid" ]]; then kill "$(cat "$_S/pid")" 2>/dev/null; rm -f "$_S/pid"; fi
  sleep 1; rm -f "$_S/containers.txt" "$_S/stack_alive" "$_S/stack_stale"; echo "Stopped training (mock)."
}

dr-start-evaluation() {
  local OPTIND opt
  if _mock_stack_blocked "deepracer-eval-${DR_RUN_ID:-0}"; then echo "ERROR: Processes running in stack deepracer-eval-${DR_RUN_ID:-0}. Stop evaluation with dr-stop-evaluation."; return 1; fi
  while getopts ":qc" opt; do :; done
  if [[ -f "$_S/pid" ]] && kill -0 "$(cat "$_S/pid")" 2>/dev/null; then
    echo "ERROR: Processes running in stack deepracer-0."; return 1; fi
  nohup python3 "$DR_DIR/tools/mock_trainer.py" eval >/dev/null 2>&1 &
  echo $! > "$_S/pid"; disown
  echo "Starting evaluation (mock)"
}
dr-stop-evaluation() { dr-stop-training; }

dr-stop-all() {
  echo "Removing stack: deepracer-0 (mock)"; echo "Removing stack: s3 (mock)"
  if [[ -f "$_S/pid" ]]; then kill "$(cat "$_S/pid")" 2>/dev/null; rm -f "$_S/pid"; fi
  rm -f "$_S/containers.txt" "$_S/stack_alive" "$_S/stack_stale"; echo "Waiting 10 seconds for stacks and services to stop... (mock: skipped)"
}
dr-summary() { printf '\033[1mEnvironment summary\033[0m (mock)\n'; [[ -f "$_S/containers.txt" ]] && cat "$_S/containers.txt" || echo "(no containers)"; }
dr-find-sagemaker() { [[ -f "$_S/containers.txt" ]] && grep -- "-algo-" "$_S/containers.txt" | cut -d'|' -f1; return 0; }
dr-find-robomaker() { [[ -f "$_S/containers.txt" ]] && grep "robomaker" "$_S/containers.txt" | cut -d'|' -f1; return 0; }
dr-download-custom-files() { echo "Downloading custom files (mock)"; }
dr-start-viewer() { echo "Viewer started (mock)"; }
dr-stop-viewer() { echo "Viewer stopped (mock)"; }
dr-update-viewer() { dr-stop-viewer; dr-start-viewer; }
dr-start-loganalysis() { echo "Log analysis started (mock)"; }
dr-stop-loganalysis() { echo "Log-analysis is not running."; }
dr-start-metrics() { echo "Metrics started (mock)"; }
dr-stop-metrics() { echo "Metrics stopped (mock)"; }

dr-create-car-zip() {
  local OPTIND opt ckpt="last" prefix="$DR_LOCAL_S3_MODEL_PREFIX" out=""
  while getopts ":bp:o:h" opt; do case $opt in b) ckpt="best" ;; p) prefix="$OPTARG" ;; o) out="$OPTARG" ;; esac; done
  [[ -z "$out" ]] && out="$DR_DIR/data/output/$prefix.tar.gz"
  local mdir="$_BUCKET/$prefix/model"
  [[ -d "$mdir" ]] || { echo "No model found for prefix $prefix" >&2; return 1; }
  local w; w="$(mktemp -d)"; mkdir -p "$w/agent"
  echo "mock frozen graph ($ckpt checkpoint)" > "$w/agent/model.pb"
  cp "$mdir/model_metadata.json" "$w/model_metadata.json" 2>/dev/null || echo '{}' > "$w/model_metadata.json"
  mkdir -p "$(dirname "$out")"; tar -czf "$out" -C "$w" agent model_metadata.json; rm -rf "$w"
  echo "Created $out"
}
