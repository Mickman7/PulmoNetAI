"""
Evaluates generated reports against the benchmark set's ground truth.

Split deliberately into TWO kinds of checks:

1. DETERMINISTIC (no LLM involved, no ambiguity, cheap, always reproducible):
   - Template Adherence: does the report contain every required section header?
   - Guideline Attribution Rate: does it contain a [Source: ...] citation pattern?
   - Critical Omission Rate: for cases with KNOWN red flags (from benchmark
     ground truth), does the report actually mention them?

2. LLM-AS-JUDGE (uses gpt-4o-mini to score, since these require semantic
   judgement a regex can't do):
   - Faithfulness: are the report's claims actually grounded in the input
     data, or does it state things the input never said?
   - Diagnostic Alignment: does the report's stated diagnostic direction
     match the case's expected_label?

Requires: benchmark_cases.json (run generate_benchmark.py first) and a real
OPENAI_API_KEY in your environment (the graph calls real LLMs).

I could not run this end-to-end myself (no working OpenAI key / no working
torch in my sandbox to run your actual model+agent). Run it on your machine.
"""

import json
import os
import re
import sys

import pandas as pd
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))  # repo root, so `backend.*` imports resolve

from backend.agent.graph import agent as pulmo_agent

BENCHMARK_PATH = os.path.join(os.path.dirname(__file__), "benchmark_cases.json")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results")
TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "..", "agent", "report_template.txt")

judge_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ---------------------------------------------------------------------------
# 1. Deterministic checks -- no LLM, no ambiguity
# ---------------------------------------------------------------------------
def check_template_adherence(report_text: str) -> bool:
    """Extracts the required section headers from report_template.txt
    (lines starting with '##' or numbered like '1. SECTION') and checks
    every one of them actually appears in the generated report."""
    with open(TEMPLATE_PATH) as f:
        template = f.read()

    required_headers = re.findall(r"^(?:#{1,3}\s+(.+)|(\d+\.\s+[A-Z][A-Z\s&/]+))$", template, re.MULTILINE)
    required_headers = [h[0] or h[1] for h in required_headers if h[0] or h[1]]

    if not required_headers:
        return True  # nothing to check against -- don't fail silently on a template with no detectable headers

    return all(header.strip() in report_text for header in required_headers)


def check_guideline_attribution(report_text: str) -> bool:
    """Looks for a citation pattern like [Source: NG250] or [Source: ...]."""
    return bool(re.search(r"\[Source:\s*[^\]]+\]", report_text, re.IGNORECASE))


def check_critical_omissions(report_text: str, red_flags: list[str]) -> tuple[bool, list[str]]:
    """For each known red flag in this case's ground truth, checks whether
    the NUMBER involved actually appears somewhere in the report. Returns
    (all_mentioned, list_of_missed_flags)."""
    if not red_flags:
        return True, []  # no red flags in this case -- nothing to omit

    missed = []
    for flag in red_flags:
        numbers_in_flag = re.findall(r"\d+\.?\d*", flag)
        # The first number is always the patient's actual value (see generate_benchmark.py);
        # later numbers are generic guideline thresholds (e.g. "threshold >100 mg/L") that
        # show up in reports regardless of whether the patient's own value was stated --
        # crediting a match against those would let real omissions pass silently.
        patient_value = numbers_in_flag[:1]
        if patient_value and patient_value[0] not in report_text:
            missed.append(flag)

    return len(missed) == 0, missed


# ---------------------------------------------------------------------------
# 2. LLM-as-judge checks -- need semantic judgement
# ---------------------------------------------------------------------------
class FaithfulnessJudgement(BaseModel):
    is_faithful: bool = Field(description="True if every claim in the report is supported by the source data")
    unsupported_claims: list[str] = Field(description="Any claims found in the report NOT supported by the source data, empty list if none")


class DiagnosticAlignmentJudgement(BaseModel):
    matches_expected: bool = Field(description="True if the report's stated diagnostic direction matches the expected label")
    stated_direction: str = Field(description="What diagnostic direction the report actually states (e.g. 'Pneumonia', 'Normal', 'unclear')")


def judge_faithfulness(report_text: str, source_data: str) -> FaithfulnessJudgement:
    structured_llm = judge_llm.with_structured_output(FaithfulnessJudgement)
    prompt = f"""You are checking whether a clinical report's claims are actually
supported by the source data it was generated from. Flag ANY claim in the
report that states a specific fact, value, or finding not present in the
source data below. Do not flag reasonable clinical interpretation of the
data as unsupported -- only flag claims that contradict or invent beyond it.

The report is REQUIRED to write exactly "Not available/Not assessed" for any
field the source data doesn't cover -- that literal placeholder is the
correct, mandated behavior for missing data, never an invented claim. Do NOT
flag "Not available/Not assessed" (or trivial rewordings of it, e.g. "not
documented", "not provided") as unsupported.

The report may also state values it computed from the source data rather than
copying verbatim -- most commonly a patient's age, computed from their DOB
and the assessment date. Before flagging a computed value as unsupported, do
the computation yourself from the source data given below. Only flag it if
the computation is actually wrong; a correctly-computed value is faithful
even though that exact number never appears literally in the source data.
The same applies to lab values (WBC, CRP, etc.) restated from a specific
record above -- check the source data's own numbers before flagging a
restated value as invented.

--- Source data ---
{source_data}

--- Report to check ---
{report_text}"""
    return structured_llm.invoke(prompt)


def judge_diagnostic_alignment(report_text: str, expected_label: str) -> DiagnosticAlignmentJudgement:
    structured_llm = judge_llm.with_structured_output(DiagnosticAlignmentJudgement)
    prompt = f"""Read this clinical report and determine what diagnostic direction
it ultimately states or supports (Pneumonia / Normal / unclear).
Expected direction for this case: {expected_label}

--- Report ---
{report_text}"""
    return structured_llm.invoke(prompt)


# ---------------------------------------------------------------------------
# 3. Run the actual agent + evaluate
# ---------------------------------------------------------------------------
def build_source_data_summary(case: dict) -> str:
    """The 'ground truth' input data a faithfulness judge checks the report against.

    Must mirror every field the report-generating agent itself receives
    (see _render_records in agent/nodes.py) -- omitting a field here doesn't
    stop the agent from citing it, it just makes the judge flag legitimately
    grounded content as invented."""
    lines = [case["patient_summary"]]
    for r in case["records"]:
        lines.append(
            f"{r['created_at']} - Prediction: {r['label']} ({r['probability']:.2f}). "
            f"Attention: {r['attention_focus']}. Notes: {r['notes']}. "
            f"WBC: {r.get('wbc')}, CRP: {r.get('crp')}. Vitals: {r.get('vitals_summary') or 'Not provided'}."
        )
    return "\n".join(lines)


def evaluate_case(case: dict) -> dict:
    initial_state = {
        "patient_summary": case["patient_summary"],
        "records": case["records"],
        "guideline_context": None,
        "analysis": None,
        "reasoning": None,
        "report": None,
    }

    final_state = pulmo_agent.invoke(initial_state)
    report_text = final_state["report"]

    source_data = build_source_data_summary(case)

    template_ok = check_template_adherence(report_text)
    attribution_ok = check_guideline_attribution(report_text)
    omission_ok, missed_flags = check_critical_omissions(report_text, case["ground_truth"]["red_flags"])
    faithfulness = judge_faithfulness(report_text, source_data)
    alignment = judge_diagnostic_alignment(report_text, case["ground_truth"]["expected_label"])

    return {
        "case_id": case["case_id"],
        "category": case["category"],
        "template_adherence": template_ok,
        "guideline_attribution": attribution_ok,
        "critical_omission_free": omission_ok,
        "missed_red_flags": "; ".join(missed_flags),
        "faithful": faithfulness.is_faithful,
        "unsupported_claims": "; ".join(faithfulness.unsupported_claims),
        "diagnostic_aligned": alignment.matches_expected,
        "stated_direction": alignment.stated_direction,
        "expected_direction": case["ground_truth"]["expected_label"],
        "report_text": report_text,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(BENCHMARK_PATH) as f:
        cases = json.load(f)
    print(f"Loaded {len(cases)} benchmark cases. Running full agent on each (this calls real LLMs)...")

    results = []
    for i, case in enumerate(cases, start=1):
        print(f"  [{i}/{len(cases)}] {case['case_id']} ({case['category']})...")
        try:
            results.append(evaluate_case(case))
        except Exception as e:
            print(f"    FAILED: {e}")
            results.append({"case_id": case["case_id"], "category": case["category"], "error": str(e)})

    df = pd.DataFrame(results)
    df.to_csv(f"{OUTPUT_DIR}/report_evaluation_raw.csv", index=False)

    n = len(df)
    summary = {
        "Faithfulness (report-level pass rate)": f"{df['faithful'].sum()}/{n} ({df['faithful'].mean():.1%})" if "faithful" in df else "N/A",
        "Diagnostic Alignment": f"{df['diagnostic_aligned'].sum()}/{n} ({df['diagnostic_aligned'].mean():.1%})" if "diagnostic_aligned" in df else "N/A",
        "Guideline Attribution Rate": f"{df['guideline_attribution'].sum()}/{n} ({df['guideline_attribution'].mean():.1%})" if "guideline_attribution" in df else "N/A",
        "Critical Omission Rate": f"{n - df['critical_omission_free'].sum()}/{n} ({(1 - df['critical_omission_free'].mean()):.1%})" if "critical_omission_free" in df else "N/A",
        "Template Adherence": f"{df['template_adherence'].sum()}/{n} ({df['template_adherence'].mean():.1%})" if "template_adherence" in df else "N/A",
    }

    print("\n" + "=" * 60)
    print(f"TABLE 5.4 -- GENERATIVE CLINICAL REPORT BENCHMARKS (n={n})")
    print("=" * 60)
    for k, v in summary.items():
        print(f"{k}: {v}")

    with open(f"{OUTPUT_DIR}/report_evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFull results saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()