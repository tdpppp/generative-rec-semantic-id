#!/usr/bin/env python3
"""Resumable, single-GPU pipeline for the ep3/history50 experiment sequence."""

import fcntl
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv/bin/python"
CHECKPOINT_ROOT = ROOT / "outputs/gr/checkpoints"
RUN_ROOT = ROOT / "outputs/gr/ep3_history50_pipeline"
ATTEMPT_ROOT = RUN_ROOT / "attempts"
STATE_PATH = RUN_ROOT / "state.json"
EVENTS_PATH = RUN_ROOT / "events.jsonl"
LOCK_PATH = RUN_ROOT / "pipeline.lock"
STOP_PATH = RUN_ROOT / "STOP"
PROFILE_PATH = ROOT / "data/TencentGR_1M/cache/profiles/10k-u10000-mm81-82"
MAPPING_DIR = ROOT / "outputs/10k_balanced/emb_infer/sinkhorn"
SOURCE_CHECKPOINT = CHECKPOINT_ROOT / "qwen25_05b_10k_ep2_lr3e5_b32/checkpoint-54000"
EP3_NAME = "qwen25_05b_10k_ep3_from_best_lr1e5_b32"
HISTORY_NAME = "qwen25_05b_10k_ep3_history50_from_best_lr1e5_b32"
EP3_DIR = CHECKPOINT_ROOT / EP3_NAME
HISTORY_DIR = CHECKPOINT_ROOT / HISTORY_NAME
EP3_CONFIG = ROOT / "gr/gr_train_qwen25_05b_10k_ep3_from_best_lr1e5_b32.json"
HISTORY_CONFIG = ROOT / "gr/gr_train_qwen25_05b_10k_ep3_history50_from_best_lr1e5_b32.json"
TOP_K = 10
EVAL_BATCH_SIZE = 8
MAX_EVAL_ATTEMPTS = 2
MAX_TRAIN_ATTEMPTS = 2
BEAM40_TIMEOUT_SECONDS = int(os.environ.get("GR_BEAM40_TIMEOUT_SECONDS", "7200"))

ENV = os.environ.copy()
ENV.update({
    "USER_CACHE_PATH": str(ROOT / "outputs"),
    "TRAIN_CKPT_PATH": str(CHECKPOINT_ROOT),
    "TRAIN_TF_EVENTS_PATH": str(ROOT / "outputs/gr/tensorboard"),
    "CUDA_VISIBLE_DEVICES": "0",
})

_legal_codes = None
_active_process = None
_stop_requested = False


class CriticalPipelineError(RuntimeError):
    pass


def now():
    return datetime.now().astimezone().isoformat()


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_state():
    if not STATE_PATH.exists():
        return {"created_at": now(), "status": "initializing", "stages": {}}
    try:
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise CriticalPipelineError(f"Invalid pipeline state file: {STATE_PATH}") from error
    state.setdefault("stages", {})
    return state


STATE = None


def save_state(status=None, current_stage=None, reason=None):
    if status is not None:
        STATE["status"] = status
    if current_stage is not None:
        STATE["current_stage"] = current_stage
    STATE["reason"] = reason
    STATE["updated_at"] = now()
    atomic_json(STATE_PATH, STATE)


def update_stage(stage, status, reason=None, **details):
    stage_state = STATE["stages"].setdefault(stage, {})
    stage_state.update(details)
    stage_state.update({"status": status, "updated_at": now(), "reason": reason})
    save_state(status="running" if status in {"running", "waiting"} else STATE.get("status"), current_stage=stage)


def record_event(stage, status, reason=None, **details):
    payload = {"time": now(), "stage": stage, "status": status, "reason": reason, **details}
    with EVENTS_PATH.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def handle_signal(signum, _frame):
    global _stop_requested
    _stop_requested = True
    stage = STATE.get("current_stage", "pipeline") if STATE is not None else "pipeline"
    record_event(stage, "stop_requested", f"received signal {signum}")
    if _active_process is not None and _active_process.poll() is None:
        os.killpg(_active_process.pid, signal.SIGTERM)


def check_stop():
    if _stop_requested or STOP_PATH.exists():
        raise CriticalPipelineError(f"Stop requested via {STOP_PATH}")


def matching_gr_processes(config_name=None):
    result = subprocess.run(
        ["ps", "-eo", "pid=,args="],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    matches = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not re.search(r"(?:^|\s)(?:\.venv/bin/)?python\S*\s+-u?\s*-m\s+gr\.(?:train_gr|evaluate_gr)\b", stripped):
            continue
        if config_name and config_name not in stripped:
            continue
        matches.append(stripped)
    return matches


def wait_for_existing_training(config_name, stage):
    announced = False
    while True:
        check_stop()
        processes = matching_gr_processes(config_name)
        if not processes:
            return announced
        if not announced:
            update_stage(stage, "waiting", "observing already-running training", processes=processes)
            record_event(stage, "waiting", "observing already-running training", processes=processes)
            announced = True
        time.sleep(30)


def wait_for_gpu_slot(stage):
    announced = False
    while True:
        check_stop()
        processes = matching_gr_processes()
        if not processes:
            return
        if not announced:
            update_stage(stage, "waiting", "another GR GPU task is active", processes=processes)
            record_event(stage, "waiting", "another GR GPU task is active", processes=processes)
            announced = True
        time.sleep(30)


def model_artifacts_complete(model_dir):
    weight_patterns = ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json", "pytorch_model.bin.index.json")
    missing = []
    if not (model_dir / "config.json").is_file():
        missing.append("config.json")
    if not any((model_dir / name).is_file() for name in weight_patterns):
        missing.append("root model weights")
    if not ((model_dir / "tokenizer.json").is_file() or (model_dir / "vocab.json").is_file()):
        missing.append("tokenizer")
    checkpoints = [path for path in model_dir.glob("checkpoint-*") if (path / "trainer_state.json").is_file()]
    if not checkpoints:
        missing.append("checkpoint trainer_state.json")
    return not missing, missing


def latest_complete_checkpoint(model_dir):
    candidates = []
    for path in model_dir.glob("checkpoint-*"):
        match = re.fullmatch(r"checkpoint-(\d+)", path.name)
        required = ("trainer_state.json", "optimizer.pt", "scheduler.pt", "config.json")
        has_weights = (path / "model.safetensors").is_file() or (path / "pytorch_model.bin").is_file()
        if match and has_weights and all((path / name).is_file() for name in required):
            candidates.append((int(match.group(1)), path))
    return max(candidates, default=(None, None))[1]


def load_legal_codes():
    global _legal_codes
    if _legal_codes is None:
        from gr.utils import load_item2token_dict
        _legal_codes = set(load_item2token_dict(MAPPING_DIR).values())
    return _legal_codes


def validate_evaluation(metrics_path, predictions_path, expected_split, expected_model, max_length, num_beams, report_path=None):
    if not metrics_path.is_file() or not predictions_path.is_file():
        raise AssertionError("metrics or predictions file is missing")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics.get("split") == expected_split, f"unexpected split: {metrics.get('split')}"
    assert metrics.get("top_k") == TOP_K, f"unexpected top_k: {metrics.get('top_k')}"
    assert metrics.get("num_beams") == num_beams, f"unexpected num_beams: {metrics.get('num_beams')}"
    assert metrics.get("max_seq_length") == max_length, f"unexpected max_seq_length: {metrics.get('max_seq_length')}"
    assert Path(metrics["model_path"]).resolve() == expected_model.resolve(), "unexpected model path"
    assert metrics["metrics"]["users"] == 10_000, f"unexpected user count: {metrics['metrics']['users']}"
    legal_codes = load_legal_codes()
    row_count = 0
    users = set()
    hits = 0
    unique_codes = set()
    with predictions_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            row_count += 1
            user_id = row.get("user_id")
            assert user_id is not None, f"line {line_number}: null user"
            users.add(user_id)
            candidates = row.get("predicted_semantic_ids")
            assert isinstance(candidates, list) and len(candidates) == TOP_K, f"line {line_number}: invalid candidates"
            assert all(code is not None for code in candidates), f"line {line_number}: null candidate"
            assert len(set(candidates)) == TOP_K, f"line {line_number}: duplicate candidate"
            assert all(code in legal_codes for code in candidates), f"line {line_number}: illegal Semantic ID"
            unique_codes.update(candidates)
            target = row.get("target_semantic_id")
            expected_rank = next((index + 1 for index, code in enumerate(candidates) if code == target), None)
            assert row.get("rank") == expected_rank, f"line {line_number}: rank mismatch"
            hits += int(expected_rank is not None)
    assert row_count == 10_000, f"expected 10000 rows, got {row_count}"
    assert len(users) == 10_000, f"expected 10000 unique users, got {len(users)}"
    assert metrics["metrics"]["hits"] == hits, f"metrics hits do not match recomputed hits ({hits})"
    assert metrics.get("valid_format_rate") == 1.0, "valid_format_rate is not 1.0"
    assert metrics.get("valid_mapping_rate") == 1.0, "valid_mapping_rate is not 1.0"
    report = {
        "passed": True,
        "split": expected_split,
        "prediction_rows": row_count,
        "unique_users": len(users),
        "candidates_per_user": TOP_K,
        "null_candidates": 0,
        "users_with_duplicate_candidates": 0,
        "illegal_candidates": 0,
        "recomputed_hits": hits,
        "metrics_hits": metrics["metrics"]["hits"],
        "unique_recommended_semantic_ids": len(unique_codes),
        "valid_format_rate": metrics["valid_format_rate"],
        "valid_mapping_rate": metrics["valid_mapping_rate"],
    }
    if report_path is not None:
        atomic_json(report_path, report)
    return metrics


def run_command(command, log_path, timeout=None):
    global _active_process
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{now()}] COMMAND: {' '.join(map(str, command))}\n")
        log.flush()
        _active_process = subprocess.Popen(
            [str(value) for value in command],
            cwd=ROOT,
            env=ENV,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            return _active_process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(_active_process.pid, signal.SIGTERM)
            try:
                _active_process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(_active_process.pid, signal.SIGKILL)
                _active_process.wait()
            return 124
        finally:
            _active_process = None


def evaluation_paths(model_dir, stem):
    return (
        model_dir / f"{stem}_metrics.json",
        model_dir / f"{stem}_predictions.jsonl",
        model_dir / f"{stem}_artifact_check.json",
        model_dir / f"evaluate_{stem}.log",
    )


def evaluation_stage(stage, model_dir, split, max_length, num_beams, stem, optional=False, timeout=None):
    metrics_path, predictions_path, report_path, log_path = evaluation_paths(model_dir, stem)
    if metrics_path.exists() and predictions_path.exists():
        try:
            metrics = validate_evaluation(
                metrics_path, predictions_path, split, model_dir, max_length, num_beams, report_path
            )
        except Exception as error:
            raise CriticalPipelineError(
                f"Existing artifacts for {stage} are incomplete or invalid and will not be overwritten: {error}"
            ) from error
        update_stage(stage, "completed", "validated existing artifacts", metrics_path=str(metrics_path))
        record_event(stage, "skipped", "validated existing artifacts")
        return metrics

    attempts = int(STATE["stages"].get(stage, {}).get("attempts", 0))
    reason = STATE["stages"].get(stage, {}).get("reason") or "attempt limit was already reached"
    attempt_parent = ATTEMPT_ROOT / stage
    for attempt_dir in sorted(attempt_parent.glob("attempt-*"), reverse=True):
        attempt_metrics = attempt_dir / "metrics.json"
        attempt_predictions = attempt_dir / "predictions.jsonl"
        attempt_report = attempt_dir / "artifact_check.json"
        try:
            metrics = validate_evaluation(
                attempt_metrics, attempt_predictions, split, model_dir, max_length, num_beams, attempt_report
            )
        except Exception:
            continue
        for source, destination in (
            (attempt_predictions, predictions_path),
            (attempt_report, report_path),
            (attempt_metrics, metrics_path),
        ):
            if destination.exists():
                continue
            shutil.copy2(source, destination)
        metrics = validate_evaluation(
            metrics_path, predictions_path, split, model_dir, max_length, num_beams, report_path
        )
        update_stage(stage, "completed", "recovered validated attempt artifacts", attempts=attempts)
        record_event(stage, "recovered", "recovered validated attempt artifacts", attempt_dir=str(attempt_dir))
        return metrics

    if metrics_path.exists() or predictions_path.exists():
        raise CriticalPipelineError(
            f"Partial canonical artifacts exist for {stage} and no validated attempt can recover them"
        )

    while attempts < MAX_EVAL_ATTEMPTS:
        check_stop()
        wait_for_gpu_slot(stage)
        attempts += 1
        attempt_dir = ATTEMPT_ROOT / stage / f"attempt-{attempts}-{datetime.now().strftime('%Y%m%dT%H%M%S')}"
        attempt_dir.mkdir(parents=True, exist_ok=False)
        attempt_metrics = attempt_dir / "metrics.json"
        attempt_predictions = attempt_dir / "predictions.jsonl"
        attempt_report = attempt_dir / "artifact_check.json"
        update_stage(stage, "running", attempts=attempts, attempt_dir=str(attempt_dir))
        record_event(stage, "started", attempt=attempts, max_seq_length=max_length, num_beams=num_beams)
        command = [
            PYTHON, "-u", "-m", "gr.evaluate_gr",
            "--model_path", model_dir,
            "--profile_path", PROFILE_PATH,
            "--mapping_dir", MAPPING_DIR,
            "--split", split,
            "--max_seq_length", str(max_length),
            "--batch_size", str(EVAL_BATCH_SIZE),
            "--top_k", str(TOP_K),
            "--num_beams", str(num_beams),
            "--device", "cuda:0",
            "--output", attempt_metrics,
            "--predictions", attempt_predictions,
        ]
        rc = run_command(command, log_path, timeout=timeout)
        if rc == 0:
            try:
                metrics = validate_evaluation(
                    attempt_metrics, attempt_predictions, split, model_dir, max_length, num_beams, attempt_report
                )
            except Exception as error:
                reason = f"artifact validation failed: {error}"
            else:
                # Write metrics last so its presence acts as the completion marker.
                for source, destination in (
                    (attempt_predictions, predictions_path),
                    (attempt_report, report_path),
                    (attempt_metrics, metrics_path),
                ):
                    if destination.exists():
                        raise CriticalPipelineError(f"Refusing to overwrite {destination}")
                    shutil.copy2(source, destination)
                update_stage(stage, "completed", attempts=attempts, metrics_path=str(metrics_path))
                record_event(stage, "completed", attempt=attempts, metrics_path=str(metrics_path))
                return metrics
        else:
            reason = "evaluation timed out" if rc == 124 else f"evaluation exited with status {rc}"
        update_stage(stage, "retry_pending", reason, attempts=attempts)
        record_event(stage, "attempt_failed", reason, attempt=attempts)
        if attempts < MAX_EVAL_ATTEMPTS:
            time.sleep(30)
    if optional:
        update_stage(stage, "skipped_after_failure", reason, attempts=attempts)
        record_event(stage, "skipped_after_failure", reason, attempts=attempts)
        return None
    raise CriticalPipelineError(f"Required evaluation {stage} failed after {attempts} attempts: {reason}")


def fatal_training_failure(log_path):
    if not log_path.exists():
        return False
    text = log_path.read_text(encoding="utf-8", errors="replace")[-200_000:]
    pattern = re.compile(
        r"CUDA out of memory|OutOfMemoryError|No space left on device|Disk quota exceeded|"
        r"FileNotFoundError|ModuleNotFoundError|ImportError|invalid.*checkpoint|size mismatch|data loss",
        re.IGNORECASE,
    )
    return bool(pattern.search(text))


def make_resume_config(checkpoint):
    payload = json.loads(HISTORY_CONFIG.read_text(encoding="utf-8"))
    payload["training_args"]["resume_from_checkpoint"] = str(checkpoint)
    path = RUN_ROOT / f"history50_resume_{checkpoint.name}.json"
    atomic_json(path, payload)
    return path


def history_training_stage():
    stage = "history50_training"
    if wait_for_existing_training(HISTORY_CONFIG.name, stage):
        complete, missing = model_artifacts_complete(HISTORY_DIR)
        if complete:
            update_stage(stage, "completed", "observed external training completion")
            record_event(stage, "completed", "observed external training completion")
            return
        reason = f"observed training exited without complete artifacts: {missing}"
        record_event(stage, "observed_process_failed", reason)
        if fatal_training_failure(HISTORY_DIR / "train.log"):
            raise CriticalPipelineError(f"history50 training hit a non-retriable error: {reason}")

    complete, missing = model_artifacts_complete(HISTORY_DIR)
    if complete:
        update_stage(stage, "completed", "validated existing model artifacts")
        record_event(stage, "skipped", "validated existing model artifacts")
        return

    attempts = int(STATE["stages"].get(stage, {}).get("attempts", 0))
    train_log = HISTORY_DIR / "train.log"
    while attempts < MAX_TRAIN_ATTEMPTS:
        check_stop()
        wait_for_gpu_slot(stage)
        attempts += 1
        if attempts == 1 and (not HISTORY_DIR.exists() or not any(HISTORY_DIR.iterdir())):
            config = HISTORY_CONFIG
            resume_checkpoint = None
        else:
            resume_checkpoint = latest_complete_checkpoint(HISTORY_DIR)
            if resume_checkpoint is None:
                raise CriticalPipelineError(
                    "history50 output is incomplete and has no complete checkpoint; refusing to restart over it"
                )
            config = make_resume_config(resume_checkpoint)
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        update_stage(
            stage,
            "running",
            attempts=attempts,
            config=str(config),
            resume_checkpoint=str(resume_checkpoint) if resume_checkpoint else None,
        )
        record_event(
            stage,
            "started",
            attempt=attempts,
            resume_checkpoint=str(resume_checkpoint) if resume_checkpoint else None,
        )
        rc = run_command([PYTHON, "-u", "-m", "gr.train_gr", "--config", config], train_log)
        complete, missing = model_artifacts_complete(HISTORY_DIR)
        if rc == 0 and complete:
            update_stage(stage, "completed", attempts=attempts)
            record_event(stage, "completed", attempt=attempts)
            return
        reason = f"training exited with status {rc}; missing artifacts: {missing}"
        update_stage(stage, "retry_pending", reason, attempts=attempts)
        record_event(stage, "attempt_failed", reason, attempt=attempts)
        if fatal_training_failure(train_log):
            raise CriticalPipelineError(f"history50 training hit a non-retriable error: {reason}")
        if attempts < MAX_TRAIN_ATTEMPTS:
            checkpoint = latest_complete_checkpoint(HISTORY_DIR)
            if checkpoint is None:
                raise CriticalPipelineError(f"history50 training cannot resume safely: {reason}")
            time.sleep(30)
    raise CriticalPipelineError(f"history50 training failed after {attempts} attempts")


def metric_sort_key(metrics):
    values = metrics["metrics"]
    return values["ndcg"], values["hr"], values["mrr"]


def preflight():
    stage = "preflight"
    required = [PYTHON, PROFILE_PATH, MAPPING_DIR, SOURCE_CHECKPOINT, EP3_CONFIG, HISTORY_CONFIG]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise CriticalPipelineError(f"Missing preflight paths: {missing}")
    config = json.loads(HISTORY_CONFIG.read_text(encoding="utf-8"))
    assert config["data_args"]["max_seq_length"] == 50
    assert config["training_args"]["output_dir"] == HISTORY_NAME
    assert config["model_args"]["checkpoint_path"].endswith("checkpoint-54000")
    update_stage(stage, "completed")
    record_event(stage, "completed")


def wait_for_ep3():
    stage = "ep3_training_observer"
    wait_for_existing_training(EP3_CONFIG.name, stage)
    complete, missing = model_artifacts_complete(EP3_DIR)
    if not complete:
        raise CriticalPipelineError(f"qwen_ep3 stopped without complete root artifacts: {missing}")
    update_stage(stage, "completed", "training finished and artifacts are complete")
    record_event(stage, "completed", "training finished and artifacts are complete")


def generate_summary(cross_results, beam_results, best_cross, best_beam, final_test):
    summary = {
        "generated_at": now(),
        "selection_rule": "validation NDCG@10, then HR@10, then MRR@10; descending",
        "best_cross_validation": best_cross,
        "best_beam_validation": best_beam,
        "cross_validation_results": cross_results,
        "beam_validation_results": beam_results,
        "final_test": final_test,
        "references": {
            "original_qwen_validation": {"hr": 0.0125, "ndcg": 0.007394, "mrr": 0.005841, "hits": 125},
            "original_qwen_test": {"hr": 0.0113, "ndcg": 0.006899, "mrr": 0.005565, "hits": 113},
            "sasrec_test_item_level": {"hr": 0.0052, "ndcg": 0.002697, "mrr": 0.001943},
        },
        "metric_scope_note": "Qwen results are Semantic-ID-level; SASRec is an Item-ID-level baseline.",
    }
    atomic_json(RUN_ROOT / "final_summary.json", summary)
    lines = [
        "# Ep3/history50 experiment summary",
        "",
        f"Generated: {summary['generated_at']}",
        "",
        "## Cross-validation comparison",
        "",
        "| Stage | Model | Train history | Eval history | HR@10 | NDCG@10 | MRR@10 | Hits | Unique Semantic IDs |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for entry in cross_results:
        metrics = entry["result"]["metrics"]
        lines.append(
            f"| {entry['stage']} | {entry['model_path']} | {entry['train_history']} | {entry['eval_history']} | "
            f"{metrics['hr']:.8f} | {metrics['ndcg']:.8f} | {metrics['mrr']:.8f} | {metrics['hits']} | "
            f"{entry['result']['unique_recommended_semantic_ids']} |"
        )
    lines.extend([
        "",
        "## Beam comparison",
        "",
        "| Beams | Top-k | Status | HR@10 | NDCG@10 | MRR@10 | Runtime (s) | Valid format | Valid mapping | Peak CUDA bytes |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for entry in beam_results:
        result = entry.get("result")
        if result is None:
            lines.append(f"| {entry['num_beams']} | 10 | skipped_after_failure | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
            continue
        metrics = result["metrics"]
        lines.append(
            f"| {entry['num_beams']} | 10 | completed | {metrics['hr']:.8f} | {metrics['ndcg']:.8f} | "
            f"{metrics['mrr']:.8f} | {result['evaluation_runtime_seconds']:.2f} | "
            f"{result['valid_format_rate']:.6f} | {result['valid_mapping_rate']:.6f} | "
            f"{result.get('max_cuda_memory_allocated_bytes')} |"
        )
    test_metrics = final_test["metrics"]
    lines.extend([
        "",
        "## Final test",
        "",
        f"Selected model: {best_beam['model_path']}",
        f"Evaluation history length: {best_beam['eval_history']}; num_beams: {best_beam['num_beams']}; top_k: 10.",
        f"Semantic-ID HR@10/NDCG@10/MRR@10: {test_metrics['hr']:.8f}/{test_metrics['ndcg']:.8f}/{test_metrics['mrr']:.8f}.",
        "Original Qwen Semantic-ID test HR@10/NDCG@10/MRR@10: 0.0113/0.006899/0.005565.",
        "SASRec Item-ID test HR@10/NDCG@10/MRR@10: 0.0052/0.002697/0.001943; this is not a strict same-scope comparison.",
        "Semantic-ID collisions remain a limitation when interpreting Qwen metrics.",
    ])
    (RUN_ROOT / "final_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def pipeline():
    preflight()
    wait_for_ep3()

    cross_specs = [
        ("ep3_validation_maxlen100", EP3_DIR, 100, 100, "validation_maxlen100"),
        ("ep3_validation_maxlen50", EP3_DIR, 100, 50, "validation_maxlen50"),
    ]
    cross_results = []
    for stage, model_dir, train_history, eval_history, stem in cross_specs:
        result = evaluation_stage(stage, model_dir, "validation", eval_history, 10, stem)
        cross_results.append({
            "stage": stage,
            "model_path": str(model_dir),
            "train_history": train_history,
            "eval_history": eval_history,
            "result": result,
        })

    history_training_stage()
    for stage, eval_history, stem in (
        ("history50_validation_maxlen100", 100, "validation_maxlen100"),
        ("history50_validation_maxlen50", 50, "validation_maxlen50"),
    ):
        result = evaluation_stage(stage, HISTORY_DIR, "validation", eval_history, 10, stem)
        cross_results.append({
            "stage": stage,
            "model_path": str(HISTORY_DIR),
            "train_history": 50,
            "eval_history": eval_history,
            "result": result,
        })

    best_cross = max(cross_results, key=lambda entry: metric_sort_key(entry["result"]))
    STATE["best_cross_validation"] = {
        key: value for key, value in best_cross.items() if key != "result"
    }
    save_state(current_stage="cross_validation_selection")
    record_event("cross_validation_selection", "completed", selected=STATE["best_cross_validation"])

    selected_model = Path(best_cross["model_path"])
    selected_length = best_cross["eval_history"]
    beam_results = []
    for num_beams in (10, 20, 40):
        stage = f"beam_validation_{num_beams}"
        result = evaluation_stage(
            stage,
            selected_model,
            "validation",
            selected_length,
            num_beams,
            f"validation_beams{num_beams}",
            optional=num_beams in {20, 40},
            timeout=BEAM40_TIMEOUT_SECONDS if num_beams == 40 else None,
        )
        beam_results.append({
            "stage": stage,
            "model_path": str(selected_model),
            "eval_history": selected_length,
            "num_beams": num_beams,
            "result": result,
        })

    successful_beams = [entry for entry in beam_results if entry["result"] is not None]
    if not successful_beams:
        raise CriticalPipelineError("No beam-search validation completed successfully")
    best_beam = max(successful_beams, key=lambda entry: metric_sort_key(entry["result"]))
    STATE["best_beam_validation"] = {
        key: value for key, value in best_beam.items() if key != "result"
    }
    save_state(current_stage="beam_selection")
    record_event("beam_selection", "completed", selected=STATE["best_beam_validation"])

    final_test = evaluation_stage(
        "final_test",
        Path(best_beam["model_path"]),
        "test",
        best_beam["eval_history"],
        best_beam["num_beams"],
        "test",
    )
    generate_summary(cross_results, beam_results, best_cross, best_beam, final_test)


def main():
    global STATE
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    ATTEMPT_ROOT.mkdir(parents=True, exist_ok=True)
    EVENTS_PATH.touch(exist_ok=True)
    lock_stream = LOCK_PATH.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"Another pipeline instance holds {LOCK_PATH}", file=sys.stderr)
        return 73
    lock_stream.write(str(os.getpid()))
    lock_stream.flush()
    STATE = load_state()
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    save_state(status="running", current_stage=STATE.get("current_stage", "startup"))
    record_event("pipeline", "started", pid=os.getpid())
    try:
        pipeline()
    except CriticalPipelineError as error:
        save_state(status="blocked", current_stage=STATE.get("current_stage", "pipeline"), reason=str(error))
        record_event(STATE.get("current_stage", "pipeline"), "blocked", str(error))
        return 20
    except Exception as error:
        reason = f"unexpected internal error: {error}"
        save_state(status="restart_pending", current_stage=STATE.get("current_stage", "pipeline"), reason=reason)
        record_event(STATE.get("current_stage", "pipeline"), "internal_error", reason, traceback=traceback.format_exc())
        return 75
    save_state(status="completed", current_stage="completed", reason=None)
    record_event("pipeline", "completed", summary=str(RUN_ROOT / "final_summary.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
