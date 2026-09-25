#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT_DIR/.venv/bin/python"
CHECKPOINT_ROOT="$ROOT_DIR/outputs/gr/checkpoints"
TENSORBOARD_ROOT="$ROOT_DIR/outputs/gr/tensorboard"
RUN_ROOT="$ROOT_DIR/outputs/gr/overnight"
PROFILE_PATH="$ROOT_DIR/data/TencentGR_1M/cache/profiles/10k-u10000-mm81-82"
MAPPING_DIR="$ROOT_DIR/outputs/10k_balanced/emb_infer/sinkhorn"
MODEL_PATH="$ROOT_DIR/outputs/models/Qwen2.5-0.5B"
QUEUE_LOG="$RUN_ROOT/queue.log"
EVENTS_FILE="$RUN_ROOT/queue_events.jsonl"
QUEUE_STATUS="$RUN_ROOT/queue_status.json"

export USER_CACHE_PATH="$ROOT_DIR/outputs"
export TRAIN_CKPT_PATH="$CHECKPOINT_ROOT"
export TRAIN_TF_EVENTS_PATH="$TENSORBOARD_ROOT"
export CUDA_VISIBLE_DEVICES=0

EXPERIMENTS=(
    qwen25_05b_10k_ep2_lr3e5_b32
    qwen25_05b_10k_ep2_lr5e5_b32
    qwen25_05b_10k_ep3_lr3e5_b32
    qwen25_05b_10k_ep3_lr5e5_b32
)

CONFIGS=(
    gr/gr_train_qwen25_05b_10k_ep2_lr3e5_b32.json
    gr/gr_train_qwen25_05b_10k_ep2_lr5e5_b32.json
    gr/gr_train_qwen25_05b_10k_ep3_lr3e5_b32.json
    gr/gr_train_qwen25_05b_10k_ep3_lr5e5_b32.json
)

mkdir -p "$RUN_ROOT"
touch "$QUEUE_LOG" "$EVENTS_FILE"
exec > >(tee -a "$QUEUE_LOG") 2>&1

timestamp() {
    date --iso-8601=seconds
}

record_event() {
    local experiment="$1"
    local phase="$2"
    local status="$3"
    local exit_code="$4"
    local reason="$5"
    "$PYTHON" - "$EVENTS_FILE" "$experiment" "$phase" "$status" "$exit_code" "$reason" <<'PY'
import json
import sys
from datetime import datetime

path, experiment, phase, status, exit_code, reason = sys.argv[1:]
event = {
    "time": datetime.now().astimezone().isoformat(),
    "experiment": experiment or None,
    "phase": phase,
    "status": status,
    "exit_code": int(exit_code) if exit_code else None,
    "reason": reason or None,
}
with open(path, "a", encoding="utf-8") as stream:
    stream.write(json.dumps(event, ensure_ascii=False) + "\n")
PY
}

write_experiment_status() {
    local output_dir="$1"
    local experiment="$2"
    local phase="$3"
    local status="$4"
    local started_at="$5"
    local ended_at="$6"
    local exit_code="$7"
    local reason="$8"
    "$PYTHON" - "$output_dir/experiment_status.json" "$experiment" "$phase" "$status" \
        "$started_at" "$ended_at" "$exit_code" "$reason" <<'PY'
import json
import os
import sys
import tempfile

path, experiment, phase, status, started_at, ended_at, exit_code, reason = sys.argv[1:]
payload = {
    "experiment": experiment,
    "phase": phase,
    "status": status,
    "started_at": started_at or None,
    "ended_at": ended_at or None,
    "exit_code": int(exit_code) if exit_code else None,
    "failure_reason": reason or None,
}
directory = os.path.dirname(path)
fd, temporary = tempfile.mkstemp(prefix=".experiment_status.", dir=directory, text=True)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    os.replace(temporary, path)
finally:
    if os.path.exists(temporary):
        os.unlink(temporary)
PY
}

write_queue_status() {
    local status="$1"
    local experiment="$2"
    local phase="$3"
    local reason="$4"
    "$PYTHON" - "$QUEUE_STATUS" "$status" "$experiment" "$phase" "$reason" <<'PY'
import json
import os
import sys
import tempfile
from datetime import datetime

path, status, experiment, phase, reason = sys.argv[1:]
payload = {
    "updated_at": datetime.now().astimezone().isoformat(),
    "status": status,
    "current_experiment": experiment or None,
    "phase": phase or None,
    "reason": reason or None,
}
directory = os.path.dirname(path)
fd, temporary = tempfile.mkstemp(prefix=".queue_status.", dir=directory, text=True)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    os.replace(temporary, path)
finally:
    if os.path.exists(temporary):
        os.unlink(temporary)
PY
}

classify_failure() {
    local log_path="$1"
    if grep -Eiq 'No space left on device|Disk quota exceeded|Input/output error|Stale file handle|NFS.*(error|failed)|FileNotFoundError.*(model|mapping|data|profile|train)|Required model path does not exist|Item-to-token directory does not exist|No mapping files found|ModuleNotFoundError|ImportError|CUDA (is )?(unavailable|not available)|No CUDA GPUs are available|CUDA out of memory|OutOfMemoryError|CUDA error|tensorboard.*(error|failed)|SummaryWriter.*(error|failed)' "$log_path"; then
        printf 'common'
    else
        printf 'independent'
    fi
}

validate_predictions() {
    local predictions="$1"
    local metrics="$2"
    local report="$3"
    "$PYTHON" - "$predictions" "$metrics" "$MAPPING_DIR" "$report" <<'PY'
import json
import math
import sys
from pathlib import Path

predictions_path, metrics_path, mapping_dir, report_path = map(Path, sys.argv[1:])
legal_codes = set()
for mapping_path in sorted(mapping_dir.iterdir()):
    if not mapping_path.is_file():
        continue
    with mapping_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            line = line.strip()
            if not line:
                continue
            try:
                _, code = line.split("\t", 1)
            except ValueError as error:
                raise AssertionError(f"invalid mapping line {mapping_path}:{line_number}") from error
            legal_codes.add(code)

rows = []
with predictions_path.open(encoding="utf-8") as stream:
    for line_number, line in enumerate(stream, 1):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise AssertionError(f"invalid prediction JSON at line {line_number}") from error

assert len(rows) == 10_000, f"expected 10000 prediction rows, got {len(rows)}"
user_ids = [row.get("user_id") for row in rows]
assert all(user_id is not None for user_id in user_ids), "null user_id found"
assert len(set(user_ids)) == 10_000, f"expected 10000 unique users, got {len(set(user_ids))}"

recomputed_hits = 0
for line_number, row in enumerate(rows, 1):
    assert row.get("target_item_id") is not None, f"line {line_number}: null target item ID"
    candidates = row.get("predicted_semantic_ids")
    assert isinstance(candidates, list), f"line {line_number}: candidates are not a list"
    assert len(candidates) == 10, f"line {line_number}: expected 10 candidates, got {len(candidates)}"
    assert all(code is not None for code in candidates), f"line {line_number}: null candidate found"
    assert len(set(candidates)) == 10, f"line {line_number}: duplicate candidates found"
    assert all(code in legal_codes for code in candidates), f"line {line_number}: illegal Semantic ID found"
    target = row.get("target_semantic_id")
    assert target is not None, f"line {line_number}: null target Semantic ID"
    expected_rank = next((index + 1 for index, code in enumerate(candidates) if code == target), None)
    assert row.get("rank") == expected_rank, f"line {line_number}: recorded rank does not match candidates"
    recomputed_hits += int(expected_rank is not None)

with metrics_path.open(encoding="utf-8") as stream:
    metrics = json.load(stream)
assert metrics.get("split") == "validation", f"unexpected split: {metrics.get('split')}"
assert metrics.get("top_k") == 10, f"unexpected top_k: {metrics.get('top_k')}"
assert metrics["metrics"]["users"] == 10_000, f"unexpected metrics users: {metrics['metrics']['users']}"
assert metrics["metrics"]["hits"] == recomputed_hits, (
    f"metrics hits {metrics['metrics']['hits']} != recomputed hits {recomputed_hits}"
)
assert math.isclose(metrics.get("valid_format_rate", 0.0), 1.0, abs_tol=0.0), "valid_format_rate is not 1.0"
assert math.isclose(metrics.get("valid_mapping_rate", 0.0), 1.0, abs_tol=0.0), "valid_mapping_rate is not 1.0"

report = {
    "passed": True,
    "prediction_rows": len(rows),
    "unique_users": len(set(user_ids)),
    "candidates_per_user": 10,
    "null_candidates": 0,
    "users_with_duplicate_candidates": 0,
    "illegal_candidates": 0,
    "recomputed_hits": recomputed_hits,
    "metrics_hits": metrics["metrics"]["hits"],
    "valid_format_rate": metrics["valid_format_rate"],
    "valid_mapping_rate": metrics["valid_mapping_rate"],
}
report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(report, ensure_ascii=False))
PY
}

generate_summary() {
    "$PYTHON" - "$CHECKPOINT_ROOT" "$RUN_ROOT" "${EXPERIMENTS[@]}" <<'PY'
import ast
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path

checkpoint_root = Path(sys.argv[1])
run_root = Path(sys.argv[2])
experiments = sys.argv[3:]

REFERENCE = {
    "qwen25_05b_10k_v1_test": {"hr": 0.0081, "recall": 0.0081, "ndcg": 0.0054992355, "mrr": 0.0046959127},
    "most_popular_test": {"hr": 0.0073, "recall": 0.0073, "ndcg": 0.0050109397, "mrr": 0.0043209524},
    "random_test": {"hr": 0.0001},
}

def directory_size(path):
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())

def parse_training_metrics(log_path):
    metrics = {}
    if not log_path.exists():
        return metrics
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "train_runtime" not in line:
            continue
        start = line.find("{")
        end = line.rfind("}")
        if start < 0 or end <= start:
            continue
        try:
            candidate = ast.literal_eval(line[start:end + 1])
        except (SyntaxError, ValueError):
            continue
        if isinstance(candidate, dict) and "train_runtime" in candidate:
            metrics = candidate
    return metrics

results = []
for experiment in experiments:
    output_dir = checkpoint_root / experiment
    status_path = output_dir / "experiment_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {"status": "missing"}
    result = {"experiment": experiment, "status": status.get("status"), "failure_reason": status.get("failure_reason")}
    metrics_path = output_dir / "validation_metrics.json"
    if metrics_path.exists():
        validation = json.loads(metrics_path.read_text(encoding="utf-8"))
        result["validation"] = validation
    state_files = sorted(output_dir.glob("checkpoint-*/trainer_state.json"))
    states = []
    for state_path in state_files:
        try:
            states.append(json.loads(state_path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            pass
    states.sort(key=lambda state: state.get("global_step", -1))
    if states:
        latest = states[-1]
        result["best_validation_loss"] = latest.get("best_metric")
        best_path = latest.get("best_model_checkpoint") or ""
        match = re.search(r"checkpoint-(\d+)$", best_path)
        result["best_validation_loss_step"] = int(match.group(1)) if match else None
    train_metrics = parse_training_metrics(output_dir / "train.log")
    result["training"] = {
        "runtime_seconds": train_metrics.get("train_runtime"),
        "samples_per_second": train_metrics.get("train_samples_per_second"),
        "steps_per_second": train_metrics.get("train_steps_per_second"),
        "epoch": train_metrics.get("epoch"),
    }
    log_text = (output_dir / "train.log").read_text(encoding="utf-8", errors="replace") if (output_dir / "train.log").exists() else ""
    result["nonfinite_loss_or_grad_norm"] = bool(re.search(
        r"['\"](?:loss|grad_norm)['\"]\s*:\s*(?:nan|[-+]?inf(?:inity)?)\b",
        log_text,
        flags=re.IGNORECASE,
    ))
    result["disk_usage_bytes"] = directory_size(output_dir) if output_dir.exists() else 0
    results.append(result)

successful = [entry for entry in results if entry.get("status") == "completed" and "validation" in entry]
successful.sort(key=lambda entry: (
    -entry["validation"]["metrics"]["ndcg"],
    -entry["validation"]["metrics"]["hr"],
    -entry["validation"]["metrics"]["mrr"],
))
recommended = successful[0]["experiment"] if successful else None
summary = {
    "generated_at": datetime.now().astimezone().isoformat(),
    "selection_rule": "validation NDCG@10, then HR@10, then MRR@10; descending",
    "recommended_experiment": recommended,
    "experiments": results,
    "reference_results": REFERENCE,
    "comparison_note": "References are test-split results supplied before this run; new experiments are validation-only and are not directly split-equivalent.",
}
(run_root / "final_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

lines = [
    "# GR overnight experiment summary",
    "",
    f"Generated: {summary['generated_at']}",
    f"Recommended by validation ranking: {recommended or 'none'}",
    "",
    "| Experiment | Status | Best eval loss (step) | HR@10 | Recall@10 | NDCG@10 | MRR@10 | Unique IDs | Coverage | Runtime (s) | Samples/s | Nonfinite | Disk bytes |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for result in results:
    validation = result.get("validation", {})
    metrics = validation.get("metrics", {})
    training = result.get("training", {})
    best = result.get("best_validation_loss")
    step = result.get("best_validation_loss_step")
    best_text = f"{best:.8f} ({step})" if isinstance(best, (int, float)) else "n/a"
    lines.append(
        f"| {result['experiment']} | {result.get('status')} | {best_text} | "
        f"{metrics.get('hr', 'n/a')} | {metrics.get('recall', 'n/a')} | {metrics.get('ndcg', 'n/a')} | "
        f"{metrics.get('mrr', 'n/a')} | {validation.get('unique_recommended_semantic_ids', 'n/a')} | "
        f"{validation.get('semantic_id_catalog_coverage', 'n/a')} | {training.get('runtime_seconds', 'n/a')} | "
        f"{training.get('samples_per_second', 'n/a')} | {result['nonfinite_loss_or_grad_norm']} | {result['disk_usage_bytes']} |"
    )
lines.extend([
    "",
    "Reference results supplied for context (test split): Qwen v1 HR/NDCG/MRR = 0.0081/0.0054992355/0.0046959127; "
    "MostPopular = 0.0073/0.0050109397/0.0043209524; Random HR = 0.0001.",
    "New model selection uses validation only; no test evaluation was run.",
])
(run_root / "final_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps({"recommended_experiment": recommended, "successful_experiments": len(successful)}))
PY
}

preflight() {
    echo "[$(timestamp)] Starting preflight checks"
    [[ -x "$PYTHON" ]] || { echo "Python is not executable: $PYTHON"; return 1; }
    [[ -d "$MODEL_PATH" ]] || { echo "Base model is missing: $MODEL_PATH"; return 1; }
    [[ -d "$PROFILE_PATH" ]] || { echo "Profile is missing: $PROFILE_PATH"; return 1; }
    [[ -d "$MAPPING_DIR" ]] || { echo "Mapping is missing: $MAPPING_DIR"; return 1; }
    command -v nvidia-smi >/dev/null
    command -v git >/dev/null

    local gpu_processes
    gpu_processes="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')"
    [[ -z "$gpu_processes" ]] || { echo "GPU already has compute processes: $gpu_processes"; return 1; }

    local experiment config output_dir
    for experiment in "${EXPERIMENTS[@]}"; do
        output_dir="$CHECKPOINT_ROOT/$experiment"
        if [[ -d "$output_dir" ]] && find "$output_dir" -mindepth 1 -print -quit | grep -q .; then
            echo "Refusing to overwrite non-empty experiment directory: $output_dir"
            return 1
        fi
    done

    cd "$ROOT_DIR"
    for config in "${CONFIGS[@]}"; do
        "$PYTHON" - "$config" <<'PY'
import sys
from gr.train_gr import parse_args_from_json

model_args, data_args, training_args = parse_args_from_json(sys.argv[1])
assert model_args.from_pretrained and not model_args.from_checkpoint
assert training_args.per_device_train_batch_size == 32
assert training_args.per_device_eval_batch_size == 32
assert training_args.bf16 and not training_args.fp16
assert training_args.load_best_model_at_end
assert data_args.max_seq_length == 100
print(f"parsed {sys.argv[1]}: epochs={training_args.num_train_epochs}, lr={training_args.learning_rate}")
PY
    done
    git diff --check
    "$PYTHON" -m unittest discover -s tests -v
    echo "[$(timestamp)] Preflight checks passed"
}

run_experiment() {
    local experiment="$1"
    local config="$2"
    local output_dir="$CHECKPOINT_ROOT/$experiment"
    local train_log="$output_dir/train.log"
    local eval_log="$output_dir/evaluate_validation.log"
    local metrics="$output_dir/validation_metrics.json"
    local predictions="$output_dir/validation_predictions.jsonl"
    local artifact_report="$output_dir/validation_artifact_check.json"
    local started_at ended_at rc reason failure_class

    mkdir -p "$output_dir"
    started_at="$(timestamp)"
    echo "[$started_at] Starting training: $experiment"
    write_queue_status running "$experiment" training ""
    write_experiment_status "$output_dir" "$experiment" training running "$started_at" "" "" ""
    record_event "$experiment" training started "" ""

    set +e
    "$PYTHON" -m gr.train_gr --config "$ROOT_DIR/$config" 2>&1 | tee "$train_log"
    rc=${PIPESTATUS[0]}
    set -e
    if (( rc != 0 )); then
        ended_at="$(timestamp)"
        reason="training exited with status $rc"
        failure_class="$(classify_failure "$train_log")"
        echo "[$ended_at] $experiment failed: $reason ($failure_class)"
        write_experiment_status "$output_dir" "$experiment" training failed "$started_at" "$ended_at" "$rc" "$reason; class=$failure_class"
        record_event "$experiment" training failed "$rc" "$reason; class=$failure_class"
        [[ "$failure_class" == common ]] && return 20
        return 10
    fi

    if [[ ! -s "$output_dir/config.json" ]] || ! find "$output_dir" -maxdepth 1 -type f \( -name 'model.safetensors' -o -name 'pytorch_model.bin' \) -print -quit | grep -q .; then
        ended_at="$(timestamp)"
        reason="training returned success but the root best-model artifact is incomplete"
        write_experiment_status "$output_dir" "$experiment" training failed "$started_at" "$ended_at" "1" "$reason"
        record_event "$experiment" training failed "1" "$reason"
        return 10
    fi
    if ! find "$output_dir" -mindepth 2 -maxdepth 2 -path '*/checkpoint-*/trainer_state.json' -print -quit | grep -q .; then
        ended_at="$(timestamp)"
        reason="training returned success but no checkpoint trainer_state.json was found"
        write_experiment_status "$output_dir" "$experiment" training failed "$started_at" "$ended_at" "1" "$reason"
        record_event "$experiment" training failed "1" "$reason"
        return 10
    fi

    echo "[$(timestamp)] Training completed; starting validation Top-10: $experiment"
    write_queue_status running "$experiment" validation ""
    write_experiment_status "$output_dir" "$experiment" validation running "$started_at" "" "" ""
    record_event "$experiment" validation started "" ""
    set +e
    "$PYTHON" -m gr.evaluate_gr \
        --model_path "$output_dir" \
        --profile_path "$PROFILE_PATH" \
        --mapping_dir "$MAPPING_DIR" \
        --split validation \
        --batch_size 16 \
        --top_k 10 \
        --device cuda:0 \
        --output "$metrics" \
        --predictions "$predictions" 2>&1 | tee "$eval_log"
    rc=${PIPESTATUS[0]}
    set -e
    if (( rc != 0 )); then
        ended_at="$(timestamp)"
        reason="validation evaluation exited with status $rc"
        failure_class="$(classify_failure "$eval_log")"
        echo "[$ended_at] $experiment failed: $reason ($failure_class)"
        write_experiment_status "$output_dir" "$experiment" validation failed "$started_at" "$ended_at" "$rc" "$reason; class=$failure_class"
        record_event "$experiment" validation failed "$rc" "$reason; class=$failure_class"
        [[ "$failure_class" == common ]] && return 20
        return 10
    fi

    set +e
    validate_predictions "$predictions" "$metrics" "$artifact_report" 2>&1 | tee -a "$eval_log"
    rc=${PIPESTATUS[0]}
    set -e
    ended_at="$(timestamp)"
    if (( rc != 0 )); then
        reason="validation artifact checks failed with status $rc"
        echo "[$ended_at] $experiment failed: $reason"
        write_experiment_status "$output_dir" "$experiment" artifact_validation failed "$started_at" "$ended_at" "$rc" "$reason"
        record_event "$experiment" artifact_validation failed "$rc" "$reason"
        return 10
    fi

    write_experiment_status "$output_dir" "$experiment" completed completed "$started_at" "$ended_at" "0" ""
    record_event "$experiment" completed completed "0" ""
    echo "[$ended_at] Completed training, validation, and artifact checks: $experiment"
    return 0
}

on_unexpected_error() {
    local rc=$?
    local line="$1"
    write_queue_status failed "" internal "unexpected queue error at line $line (status $rc)"
    record_event "" queue failed "$rc" "unexpected queue error at line $line"
    exit "$rc"
}
trap 'on_unexpected_error "$LINENO"' ERR

cd "$ROOT_DIR"
write_queue_status preflight "" preflight ""
record_event "" queue started "" ""
if ! preflight; then
    write_queue_status failed "" preflight "preflight checks failed"
    record_event "" preflight failed "1" "preflight checks failed"
    exit 1
fi

write_queue_status running "" queue ""
independent_failures=0
for index in "${!EXPERIMENTS[@]}"; do
    experiment="${EXPERIMENTS[$index]}"
    config="${CONFIGS[$index]}"
    if run_experiment "$experiment" "$config"; then
        rc=0
    else
        rc=$?
    fi
    if (( rc == 20 )); then
        reason="stopped after confirmed common failure in $experiment"
        write_queue_status failed "$experiment" stopped "$reason"
        record_event "$experiment" queue stopped "$rc" "$reason"
        generate_summary
        exit "$rc"
    elif (( rc != 0 )); then
        independent_failures=$((independent_failures + 1))
        echo "[$(timestamp)] Continuing after independent failure in $experiment"
    fi
done

generate_summary
if (( independent_failures > 0 )); then
    reason="$independent_failures experiment(s) failed independently"
    write_queue_status completed_with_failures "" completed "$reason"
    record_event "" queue completed_with_failures "10" "$reason"
    exit 10
fi
write_queue_status completed "" completed ""
record_event "" queue completed "0" ""
echo "[$(timestamp)] All four experiments completed successfully"
