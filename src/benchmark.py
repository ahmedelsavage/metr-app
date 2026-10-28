
import csv
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

from stores.llm import (
    analyze_contract,
    generate_response,
)
from stores.llm.log_mlflow import log_benchmark_run


# =========================
# Configuration
# =========================

RUN_MODEL = "Qwen3-8B"
PROMPT_VERSION = "v1"
MAX_RETRIES = 2

CASES_PATH = Path(
    os.getenv(
        "BENCHMARK_CASES_PATH",
        str(BASE_DIR / "benchmark_cases.json"),
    )
)

OUTPUT_DIR = BASE_DIR / "benchmark_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Helper functions
# =========================

def normalize_label(value):
    """Normalize labels to make comparisons consistent."""
    if value is None:
        return ""

    if isinstance(value, dict):
        value = value.get("risk_type", value.get("label", ""))

    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def get_contract_text(case):
    """Support common contract-text field names."""
    for key in ("contract_text", "text", "contract", "input_text"):
        value = case.get(key)
        if isinstance(value, str) and value.strip():
            return value

    return ""


def get_expected_risks(case):
    """Read expected risk labels from a benchmark case."""
    values = case.get("expected_risks", [])

    if not isinstance(values, list):
        return set()

    return {
        normalize_label(value)
        for value in values
        if normalize_label(value)
    }


def get_predicted_risks(analysis):
    """Extract predicted risk labels from the analysis result."""
    if hasattr(analysis, "model_dump"):
        analysis = analysis.model_dump()

    if not isinstance(analysis, dict):
        raise ValueError("Analysis result is not a dictionary.")

    risks = analysis.get("risks", [])

    if not isinstance(risks, list):
        raise ValueError("The risks field is not a list.")

    labels = set()

    for risk in risks:
        if isinstance(risk, dict):
            label = risk.get("risk_type")
        else:
            label = getattr(risk, "risk_type", None)

        normalized = normalize_label(label)

        if normalized:
            labels.add(normalized)

    return labels


def get_analysis_dict(analysis):
    """Convert a Pydantic result into a regular dictionary."""
    if hasattr(analysis, "model_dump"):
        return analysis.model_dump()

    if isinstance(analysis, dict):
        return analysis

    raise ValueError("Unsupported analysis result type.")


def safe_rate(numerator, denominator):
    if denominator == 0:
        return 0.0
    return numerator / denominator


def percentile_95(values):
    if not values:
        return 0.0

    sorted_values = sorted(values)
    index = math.ceil(0.95 * len(sorted_values)) - 1
    return sorted_values[max(0, index)]


# =========================
# Evaluate one case
# =========================

def evaluate_case(case):
    case_id = case.get("id", "unknown")
    contract_text = get_contract_text(case)
    expected_risks = get_expected_risks(case)

    result = {
        "id": case_id,
        "success": False,
        "error": None,
        "latency_seconds": 0.0,
        "generate_calls": 0,
        "valid_json": False,
        "valid_schema_first_try": False,
        "recovered_by_retry": False,
        "expected_risks": sorted(expected_risks),
        "predicted_risks": [],
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "verdict_correct": False,
        "severity_correct": 0,
        "severity_total": 0,
        "analysis": None,
    }

    if not contract_text:
        result["error"] = "No contract text found in this case."
        return result

    generated_responses = []

    def tracked_generate(prompt):
        """Count model calls and retain raw responses for evaluation."""
        result["generate_calls"] += 1
        response = generate_response(prompt)
        generated_responses.append(response)
        return response

    start_time = time.perf_counter()

    try:
        analysis = analyze_contract(
            contract_text=contract_text,
            legal_context=case.get("legal_context", ""),
            language=case.get("language", "en"),
            generate_fn=tracked_generate,
            max_retries=MAX_RETRIES,
        )

        analysis_dict = get_analysis_dict(analysis)
        predicted_risks = get_predicted_risks(analysis)

        result["success"] = True
        result["analysis"] = analysis_dict
        result["predicted_risks"] = sorted(predicted_risks)

        # JSON validity: at least one response was valid JSON.
        for raw_response in generated_responses:
            try:
                if isinstance(raw_response, dict):
                    result["valid_json"] = True
                    break

                json.loads(raw_response)
                result["valid_json"] = True
                break
            except (TypeError, json.JSONDecodeError):
                continue

        # A successful result on one call means no retry was needed.
        result["valid_schema_first_try"] = (
            result["generate_calls"] == 1
        )

        result["recovered_by_retry"] = (
            result["generate_calls"] > 1
        )

        # Risk-label metrics.
        expected = expected_risks
        predicted = predicted_risks

        result["tp"] = len(expected & predicted)
        result["fp"] = len(predicted - expected)
        result["fn"] = len(expected - predicted)

        # Expected verdict: safe if there are no expected risks.
        expected_verdict = "risky" if expected else "safe"
        actual_verdict = normalize_label(
            analysis_dict.get("verdict", "")
        )

        result["verdict_correct"] = (
            actual_verdict == expected_verdict
        )

        # Compare severity only for risk types with expected severity labels.
        expected_severities = case.get("expected_severities", {})

        if isinstance(expected_severities, dict):
            predicted_severities = {}

            for risk in analysis_dict.get("risks", []):
                if isinstance(risk, dict):
                    risk_type = normalize_label(risk.get("risk_type"))
                    severity = normalize_label(risk.get("severity"))
                    if risk_type:
                        predicted_severities[risk_type] = severity

            for risk_type, expected_severity in expected_severities.items():
                risk_type = normalize_label(risk_type)
                expected_severity = normalize_label(expected_severity)

                if not expected_severity:
                    continue

                result["severity_total"] += 1

                if predicted_severities.get(risk_type) == expected_severity:
                    result["severity_correct"] += 1

    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"

    finally:
        result["latency_seconds"] = round(
            time.perf_counter() - start_time, 4
        )

    return result


# =========================
# Calculate summary metrics
# =========================

def calculate_summary(results):
    total = len(results)

    successful = sum(r["success"] for r in results)
    valid_json = sum(r["valid_json"] for r in results)
    valid_schema_first = sum(
        r["valid_schema_first_try"] for r in results
    )
    recovered = sum(r["recovered_by_retry"] for r in results)

    tp = sum(r["tp"] for r in results)
    fp = sum(r["fp"] for r in results)
    fn = sum(r["fn"] for r in results)

    precision = safe_rate(tp, tp + fp)
    recall = safe_rate(tp, tp + fn)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    clean_cases = [
        r for r in results if not r["expected_risks"]
    ]
    clean_cases_with_alarm = sum(
        bool(r["predicted_risks"]) for r in clean_cases
    )

    clean_cases_marked_safe = sum(
        r["success"]
        and not r["predicted_risks"]
        and r["analysis"] is not None
        and normalize_label(r["analysis"].get("verdict", "")) == "safe"
        for r in clean_cases
    )

    verdict_evaluable = [
        r for r in results
        if r["success"] and r["analysis"] is not None
    ]
    verdict_correct = sum(
        r["verdict_correct"] for r in verdict_evaluable
    )

    severity_correct = sum(r["severity_correct"] for r in results)
    severity_total = sum(r["severity_total"] for r in results)

    latencies = [r["latency_seconds"] for r in results]

    return {
        "model": RUN_MODEL,
        "total_cases": total,
        "request_success_rate": safe_rate(successful, total),
        "valid_json_rate": safe_rate(valid_json, total),
        "valid_schema_rate_first_try": safe_rate(
            valid_schema_first, total
        ),
        "recovered_by_retry": recovered,
        "valid_schema_rate_after_retry": safe_rate(successful, total),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "clean_case_count": len(clean_cases),
        "clean_cases_with_alarm": clean_cases_with_alarm,
        "false_alarm_rate": safe_rate(
            clean_cases_with_alarm, len(clean_cases)
        ),
        "clean_cases_marked_safe": clean_cases_marked_safe,
        "verdict_accuracy": safe_rate(
            verdict_correct, len(verdict_evaluable)
        ),
        "severity_accuracy": safe_rate(
            severity_correct, severity_total
        ),
        "severity_correct": severity_correct,
        "severity_total": severity_total,
        "mean_latency_seconds": (
            sum(latencies) / len(latencies) if latencies else 0.0
        ),
        "p95_latency_seconds": percentile_95(latencies),
        "error_count": sum(not r["success"] for r in results),
    }


# =========================
# Save result files
# =========================

def save_results(results, summary, run_timestamp):
    jsonl_path = OUTPUT_DIR / f"benchmark_{run_timestamp}.jsonl"
    csv_path = OUTPUT_DIR / f"benchmark_{run_timestamp}.csv"
    summary_path = OUTPUT_DIR / f"summary_{run_timestamp}.json"

    with jsonl_path.open("w", encoding="utf-8") as file:
        for result in results:
            file.write(
                json.dumps(result, ensure_ascii=False, default=str) + "\n"
            )

    # Keep nested fields readable in CSV by serializing them as JSON.
    if results:
        csv_rows = []

        for result in results:
            row = {}
            for key, value in result.items():
                if isinstance(value, (dict, list)):
                    row[key] = json.dumps(value, ensure_ascii=False)
                else:
                    row[key] = value
            csv_rows.append(row)

        with csv_path.open(
            "w", newline="", encoding="utf-8-sig"
        ) as file:
            writer = csv.DictWriter(
                file, fieldnames=list(csv_rows[0].keys())
            )
            writer.writeheader()
            writer.writerows(csv_rows)
    else:
        csv_path.write_text("", encoding="utf-8")

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    return jsonl_path, csv_path, summary_path


# =========================
# Main
# =========================

def main():
    if not CASES_PATH.exists():
        raise FileNotFoundError(
            f"Benchmark dataset not found: {CASES_PATH}"
        )

    with CASES_PATH.open("r", encoding="utf-8") as file:
        cases = json.load(file)

    if not isinstance(cases, list):
        raise ValueError(
            "benchmark_cases.json must contain a JSON list of cases."
        )

    # Optional case limit: python benchmark.py 2
    if len(sys.argv) > 1:
        try:
            limit = int(sys.argv[1])
            if limit < 1:
                raise ValueError
            cases = cases[:limit]
        except ValueError:
            raise ValueError(
                "The optional argument must be a positive integer, "
                "for example: python benchmark.py 2"
            )

    if not cases:
        raise ValueError("No benchmark cases found.")

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = []

    print(f"Model: {RUN_MODEL}")
    print(f"Dataset: {CASES_PATH}")
    print(f"Cases to run: {len(cases)}")
    print("-" * 50)

    for index, case in enumerate(cases, start=1):
        case_id = case.get("id", index)
        print(f"[{index}/{len(cases)}] Evaluating case {case_id}...")

        result = evaluate_case(case)
        results.append(result)

        if result["success"]:
            print(
                f"  Success | calls={result['generate_calls']} "
                f"| latency={result['latency_seconds']}s"
            )
        else:
            print(f"  Failed | {result['error']}")

    summary = calculate_summary(results)

    jsonl_path, csv_path, summary_path = save_results(
        results,
        summary,
        run_timestamp,
    )

    print("\nBenchmark summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    print("\nSaved files:")
    print(f"JSONL:   {jsonl_path}")
    print(f"CSV:     {csv_path}")
    print(f"Summary: {summary_path}")

    # Log the completed benchmark and result files to MLflow.
    log_benchmark_run(
        summary=summary,
        artifact_paths=[jsonl_path, csv_path, summary_path],
        dataset_name=CASES_PATH.name,
        prompt_version=PROMPT_VERSION,
    )

    print("Finished.")


if __name__ == "__main__":
    main()
