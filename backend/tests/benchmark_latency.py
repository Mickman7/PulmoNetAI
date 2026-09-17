"""
Phase-by-phase latency benchmark for the full two-stage pipeline:
Stage 1 (real multimodal classifier forward pass, backend/models/inference.py)
-> Stage 2 (RAG + LangGraph report generation: retrieval, analysis, reasoning,
report nodes, backend/agent/nodes.py).

Run with: python3 backend/tests/benchmark_latency.py [n_trials]

Requires benchmark_cases.json (run generate_benchmark.py first), the trained
checkpoint at backend/models/multimodal_pneumonia_model.pth, sample images
under backend/storage/patient_images/1/, and a real OPENAI_API_KEY (Stage 2
calls real LLMs).

Methodology notes:
- Stage 2's RAG Retrieval row is benchmarked under rag_mode="local" (vector
  search only) -- matching the table's "Vector Search" label. Hybrid mode's
  latency is dominated by live PubMed network time, already characterized
  separately in test_rag_evaluation.py; mixing it in here would swamp the
  system's own per-stage compute cost, which is this table's purpose.
- One warmup call (model load, embeddings client, vector store) runs before
  the timed trials and is excluded from the results -- it's a one-time
  server-startup cost, not steady-state per-request latency.
- The same real image is reused across trials (image-content-driven cost is
  not the variable under study here); patient text/lab data varies per trial
  from the real benchmark cases.
"""

import glob
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.models.inference import predict as model_predict
from backend.agent.nodes import retrieval_node, analysis_node, reasoning_node, report_node

BENCHMARK_PATH = os.path.join(os.path.dirname(__file__), "benchmark_cases.json")
IMAGE_DIR = os.path.join(os.path.dirname(__file__), "..", "storage", "patient_images", "1")

# (state key used while timing, table's "Node/Operation" label)
STAGES = [
    ("stage1_forward_pass", "Stage 1: Multimodal Model", "Multimodal Processing and Forward Pass"),
    ("stage2_vector_search", "Stage 2: RAG Retrieval", "Vector Search"),
    ("stage2_analysis", "Stage 2: Analysis Node", "Classifier Explanation Generation"),
    ("stage2_reasoning", "Stage 2: Reasoning Node", "Guideline Threshold Mapping"),
    ("stage2_report", "Stage 2: Report Node", "Structured Consultation Report"),
]


def _pick_image():
    imgs = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.jpg")))
    if not imgs:
        raise FileNotFoundError(f"No sample images found in {IMAGE_DIR}")
    return imgs[0]


def _fresh_state(case):
    return {
        "patient_summary": case["patient_summary"],
        "records": case["records"],
        "rag_mode": "local",
        "guideline_context": None,
        "analysis": None,
        "reasoning": None,
        "report": None,
    }


def run_trial(case, image_path):
    record = case["records"][0]
    timings = {}

    t0 = time.perf_counter()
    model_predict(image_path, record["notes"], record.get("wbc"), record.get("crp"))
    timings["stage1_forward_pass"] = time.perf_counter() - t0

    state = _fresh_state(case)

    t0 = time.perf_counter()
    state.update(retrieval_node(state))
    timings["stage2_vector_search"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    state.update(analysis_node(state))
    timings["stage2_analysis"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    state.update(reasoning_node(state))
    timings["stage2_reasoning"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    state.update(report_node(state))
    timings["stage2_report"] = time.perf_counter() - t0

    timings["total_pipeline"] = sum(timings.values())
    return timings


def main():
    os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    all_cases = json.load(open(BENCHMARK_PATH))
    cases = (all_cases * ((n_trials // len(all_cases)) + 1))[:n_trials]
    image_path = _pick_image()

    print(f"Warming caches (model load, embeddings client, vector store) -- excluded from results...")
    warmup_case = {
        "patient_summary": "Warmup Patient | DOB: 2000-01-01 | Sex: Male",
        "records": [{
            "created_at": "2026-01-01T00:00:00", "probability": 0.5, "label": "Normal",
            "attention_focus": "n/a", "notes": "warmup", "wbc": 10.0, "crp": 10.0, "vitals_summary": None,
        }],
    }
    model_predict(image_path, "warmup", 10.0, 10.0)
    warm_state = _fresh_state(warmup_case)
    warm_state.update(retrieval_node(warm_state))

    print(f"Running {n_trials} timed trials...\n")
    all_timings = []
    for i, case in enumerate(cases, start=1):
        print(f"  [{i}/{n_trials}] {case['case_id']}...")
        t = run_trial(case, image_path)
        all_timings.append(t)
        print(f"    " + ", ".join(f"{k}={v:.2f}s" for k, v in t.items()))

    total_mean = statistics.mean(t["total_pipeline"] for t in all_timings)

    print("\n" + "=" * 100)
    print("TABLE 00 -- PHASE BY PHASE EXECUTION LATENCY BREAKDOWN")
    print("=" * 100)
    header = f"{'Execution Stage':<26}{'Node/Operation':<34}{'Mean (s)':>10}{'Std Dev (s)':>13}{'% of Total':>12}"
    print(header)
    print("-" * len(header))
    for key, stage_label, node_label in STAGES:
        values = [t[key] for t in all_timings]
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        pct = (mean / total_mean * 100) if total_mean else 0.0
        print(f"{stage_label:<26}{node_label:<34}{mean:>10.2f}{std:>13.2f}{pct:>11.1f}%")

    total_values = [t["total_pipeline"] for t in all_timings]
    total_std = statistics.stdev(total_values) if len(total_values) > 1 else 0.0
    print(f"{'Total Pipeline':<26}{'End-to-end Execution':<34}{total_mean:>10.2f}{total_std:>13.2f}{100.0:>11.1f}%")

    with open(os.path.join(os.path.dirname(__file__), "results", "latency_benchmark_raw.json"), "w") as f:
        json.dump(all_timings, f, indent=2)
    print(f"\nRaw per-trial timings saved to backend/tests/results/latency_benchmark_raw.json")


if __name__ == "__main__":
    main()
