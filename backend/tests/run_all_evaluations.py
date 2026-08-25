"""
Master orchestrator: runs every test/evaluation suite in this project in
series and produces a consolidated summary. Each suite is independently
toggleable and independently parameterized, so this can run the full report
suite for a final write-up, or just one stage while iterating.

Suites, in run order:
  1. Fast wiring tests    -- mocked, offline, always cheap  (test_agent_pipeline.py, test_rag_modes.py)
  2. RAG retrieval eval   -- real local/pubmed/hybrid retrieval quality       (test_rag_evaluation.py)
  3. Benchmark generation -- synthetic case set with known ground truth      (generate_benchmark.py)
  4. Report benchmark     -- faithfulness/alignment/attribution/omission/template (evaluate_reports.py)
  5. Latency benchmark    -- phase-by-phase pipeline timing                  (benchmark_latency.py)

Suites 2, 4, and 5 call real LLMs (and 2 also calls the live PubMed API; 5
also loads the real ~550MB model checkpoint) -- they cost real time and
money. Nothing in this file makes an API call itself; it only shells out to
the existing suite scripts, always via the same interpreter that ran this
file, so venv/interpreter mismatches can't creep in.

Usage:
    python3 backend/tests/run_all_evaluations.py                  # everything, defaults
    python3 backend/tests/run_all_evaluations.py --skip-rag-eval  # skip one stage
    python3 backend/tests/run_all_evaluations.py --latency-trials 20 --benchmark-cases 10
    python3 backend/tests/run_all_evaluations.py --only fast,latency
"""

import argparse
import json
import os
import subprocess
import sys
import time

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(TESTS_DIR, "..", ".."))
RESULTS_DIR = os.path.join(TESTS_DIR, "results")
BENCHMARK_PATH = os.path.join(TESTS_DIR, "benchmark_cases.json")
PYTHON = sys.executable  # the interpreter that ran this file, not a system-default one

STAGE_NAMES = ["fast", "rag_eval", "report_eval", "latency"]


def _run(label, cmd, env=None):
    print("\n" + "=" * 78)
    print(f" RUNNING: {label}")
    print(f" $ {' '.join(cmd)}")
    print("=" * 78)
    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=REPO_ROOT, env={**os.environ, **(env or {})})
    elapsed = time.perf_counter() - t0
    status = "PASSED" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(f"--- {label}: {status} in {elapsed:.1f}s ---")
    return {"label": label, "returncode": result.returncode, "elapsed_s": round(elapsed, 1)}


def run_fast_tests():
    return _run(
        "Fast wiring tests (mocked, offline, free)",
        [PYTHON, "-m", "pytest", "backend/tests/test_agent_pipeline.py", "backend/tests/test_rag_modes.py", "-v"],
    )


def run_rag_eval():
    return _run(
        "RAG retrieval evaluation -- local / pubmed / hybrid (real APIs)",
        [PYTHON, "-m", "pytest", "backend/tests/test_rag_evaluation.py", "-s", "-v"],
        env={"RUN_RAG_EVAL": "1"},
    )


def run_generate_benchmark(n_per_category, seed):
    return _run(
        f"Generate benchmark cases (n={n_per_category}/category, seed={seed})",
        [PYTHON, "backend/tests/generate_benchmark.py", "--n-per-category", str(n_per_category), "--seed", str(seed)],
    )


def run_report_eval():
    return _run(
        "Generative report benchmark -- faithfulness/alignment/attribution/omission/template (real APIs)",
        [PYTHON, "backend/tests/evaluate_reports.py"],
    )


def run_latency(n_trials):
    return _run(
        f"Pipeline latency benchmark -- n={n_trials} trials (real model + real APIs)",
        [PYTHON, "backend/tests/benchmark_latency.py", str(n_trials)],
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run all PulmoNetAI test/evaluation suites in series.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--only", default=None,
        help=f"Comma-separated subset of stages to run: {','.join(STAGE_NAMES)} (default: all)",
    )
    parser.add_argument("--skip-fast", action="store_true", help="Skip the fast mocked wiring tests")
    parser.add_argument("--skip-rag-eval", action="store_true", help="Skip the real RAG retrieval evaluation")
    parser.add_argument("--skip-report-eval", action="store_true", help="Skip the generative report benchmark")
    parser.add_argument("--skip-latency", action="store_true", help="Skip the pipeline latency benchmark")
    parser.add_argument(
        "--reuse-benchmark", action="store_true",
        help="Reuse an existing benchmark_cases.json instead of regenerating it (report-eval/latency both need this file)",
    )
    parser.add_argument("--benchmark-cases", type=int, default=5, help="Cases per category when (re)generating the benchmark set")
    parser.add_argument("--benchmark-seed", type=int, default=42, help="Random seed for benchmark case generation")
    parser.add_argument("--latency-trials", type=int, default=10, help="Number of timed trials for the latency benchmark")
    args = parser.parse_args()

    selected = set(s.strip() for s in args.only.split(",")) if args.only else set(STAGE_NAMES)
    unknown = selected - set(STAGE_NAMES)
    if unknown:
        parser.error(f"Unknown stage(s) in --only: {', '.join(unknown)}. Valid: {', '.join(STAGE_NAMES)}")

    run_fast = "fast" in selected and not args.skip_fast
    run_rag = "rag_eval" in selected and not args.skip_rag_eval
    run_report = "report_eval" in selected and not args.skip_report_eval
    run_lat = "latency" in selected and not args.skip_latency
    needs_benchmark_file = run_report or run_lat

    print(f"PulmoNetAI test/evaluation run -- {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Stages: fast={run_fast} rag_eval={run_rag} report_eval={run_report} latency={run_lat}")
    if run_rag or run_report or run_lat:
        print("Note: this run will call real LLM/API endpoints and cost real time + money.")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary = []

    if run_fast:
        summary.append(run_fast_tests())

    if run_rag:
        summary.append(run_rag_eval())

    if needs_benchmark_file and (args.reuse_benchmark and os.path.exists(BENCHMARK_PATH)):
        print(f"\nReusing existing benchmark set at {BENCHMARK_PATH} (--reuse-benchmark)")
    elif needs_benchmark_file:
        summary.append(run_generate_benchmark(args.benchmark_cases, args.benchmark_seed))

    if run_report:
        summary.append(run_report_eval())

    if run_lat:
        summary.append(run_latency(args.latency_trials))

    print("\n" + "#" * 78)
    print("# SUMMARY")
    print("#" * 78)
    any_failed = False
    total_elapsed = 0.0
    for s in summary:
        status = "PASSED" if s["returncode"] == 0 else "FAILED"
        any_failed = any_failed or s["returncode"] != 0
        total_elapsed += s["elapsed_s"]
        print(f"  [{status:6}] {s['label']:<70} {s['elapsed_s']:>7.1f}s")
    print(f"\n  Total: {total_elapsed:.1f}s across {len(summary)} stage(s)")

    with open(os.path.join(RESULTS_DIR, "run_all_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nRun summary -> backend/tests/results/run_all_summary.json")
    print("Per-suite detail -> backend/tests/results/report_evaluation_{raw.csv,summary.json}, latency_benchmark_raw.json")

    sys.exit(1 if any_failed else 0)


if __name__ == "__main__":
    main()
