import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

from stores.llm import (
    analyze_contract,
    build_risk_prompt,
    generate_response,
    parse_analysis_response,
)

# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# Choose ONE: "Qwen3-8B", "Qwen3-4B", or "Nemotron"
RUN_MODEL = "Qwen3-8B"

# Names of the .env variables for each model: (URL, KEY)
MODEL_CONFIG = {
    "Qwen3-8B": ("QWEN8_API_URL", "QWEN8_API_KEY"),
    "Qwen3-4B": ("QWEN_API_URL", "MODEL_API_KEY"),
    "Nemotron": None,  # uses LLM_* variables through llm.py
}

CASES_PATH = Path(
    os.getenv(
        "BENCHMARK_CASES_PATH",
        str(BASE_DIR / "benchmark_cases.json"),
    )
)

OUTPUT_DIR = BASE_DIR / "benchmark_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REQUEST_TIMEOUT = 240
MAX_RETRIES = 2  # retries for invalid JSON/schema (model output)

# Connection retry settings (network problems only)
CONNECTION_ATTEMPTS = 3
CONNECTION_WAIT_SECONDS = 3
RETRY_STATUS_CODES = {429, 502, 503, 504}

RISK_TYPES = {
    "unilateral_change",
    "unilateral_termination",
    "unlimited_liability",
    "liability_exclusion",
    "unfair_arbitration",
    "unfair_deduction",
    "unfair_content_removal",
    "other",
}

SEVERITIES = {"low", "medium", "high"}


def read_env(name):
    """Read an env variable and remove spaces and quotes."""

    value = os.getenv(name) or ""
    return value.strip().strip('"').strip("'")


# --------------------------------------------------
# Call the model
# --------------------------------------------------


def get_generate_url():
    """Return the full /generate URL for the selected Qwen model."""

    url_name, _ = MODEL_CONFIG[RUN_MODEL]
    url = read_env(url_name)

    if not url:
        raise ValueError(f"{url_name} is missing from .env")

    url = url.rstrip("/")

    if not url.endswith("/generate"):
        url += "/generate"

    return url


def post_with_connection_retry(url, headers, payload):
    """
    POST to the server. Retry ONLY for connection drops and
    temporary server errors. Do not retry if the model already
    answered, and do not retry a long read timeout.
    """

    last_error = None

    for attempt in range(1, CONNECTION_ATTEMPTS + 1):
        try:
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=REQUEST_TIMEOUT,
            )

            if response.status_code in RETRY_STATUS_CODES:
                last_error = RuntimeError(
                    f"HTTP {response.status_code} from {response.url} | "
                    f"Server said: {response.text[:200]}"
                )
            else:
                return response

        except (requests.ConnectionError, requests.ConnectTimeout) as error:
            last_error = error

        if attempt < CONNECTION_ATTEMPTS:
            wait = CONNECTION_WAIT_SECONDS * attempt
            print(
                f"    Connection problem (attempt {attempt}/"
                f"{CONNECTION_ATTEMPTS}). Retrying in {wait}s..."
            )
            time.sleep(wait)

    raise RuntimeError(
        f"Server not reachable after {CONNECTION_ATTEMPTS} attempts. "
        f"Last error: {last_error}"
    )


def call_qwen(prompt):
    """Send a prompt to the Qwen API and return raw text."""

    _, key_name = MODEL_CONFIG[RUN_MODEL]
    api_key = read_env(key_name)

    if not api_key:
        raise ValueError(f"{key_name} is missing from .env")

    response = post_with_connection_retry(
        get_generate_url(),
        headers={
            "x-api-key": api_key,
            # Skips the ngrok free-plan warning page.
            "ngrok-skip-browser-warning": "true",
        },
        payload={"prompt": prompt},
    )

    if not response.ok:
        raise RuntimeError(
            f"HTTP {response.status_code} from {response.url} | "
            f"Server said: {response.text[:300]}"
        )

    data = response.json()

    if isinstance(data, str):
        return data.strip()

    if isinstance(data, dict):
        for key in ("text", "response", "generated_text", "output"):
            value = data.get(key)

            if isinstance(value, str) and value.strip():
                return value.strip()

        choices = data.get("choices")

        if isinstance(choices, list) and choices:
            first = choices[0]

            if isinstance(first, dict):
                message = first.get("message", {})

                if isinstance(message, dict):
                    content = message.get("content")

                    if isinstance(content, str) and content.strip():
                        return content.strip()

                content = first.get("text")

                if isinstance(content, str) and content.strip():
                    return content.strip()

    raise ValueError(
        f"Could not find generated text in API response: {str(data)[:300]}"
    )


def call_model(prompt):
    """Call the selected model."""

    if RUN_MODEL == "Nemotron":
        return generate_response(prompt)

    return call_qwen(prompt)


# --------------------------------------------------
# Load and check test cases
# --------------------------------------------------


def load_cases():
    if not CASES_PATH.exists():
        raise FileNotFoundError(f"Benchmark cases file not found: {CASES_PATH}")

    with CASES_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, dict):
        data = data.get("cases", [])

    if not isinstance(data, list) or not data:
        raise ValueError("The cases file must contain a list of cases.")

    return data


def get_case_id(case, index=0):
    return str(case.get("case_id", case.get("id", f"case_{index}")))


def get_expected_risks(case):
    """Return a sorted list of expected risk labels."""

    expected = case.get("expected_risks", [])

    if expected is None:
        return []

    if not isinstance(expected, list):
        raise ValueError(
            f"expected_risks must be a list in case {get_case_id(case)}"
        )

    labels = []

    for item in expected:
        if isinstance(item, str):
            label = item
        elif isinstance(item, dict):
            label = item.get("label") or item.get("risk_type") or item.get("category")
        else:
            raise ValueError(f"Invalid risk format in case {get_case_id(case)}")

        if not isinstance(label, str):
            raise ValueError(f"Missing risk label in case {get_case_id(case)}")

        label = label.strip().lower()

        if label not in RISK_TYPES:
            raise ValueError(
                f"Unknown risk label '{label}' in case {get_case_id(case)}"
            )

        labels.append(label)

    return sorted(set(labels))


def get_expected_severities(case):
    """Return {label: severity} from the case file."""

    severities = {}

    for item in case.get("expected_risks", []) or []:
        if not isinstance(item, dict):
            continue

        label = item.get("label") or item.get("risk_type") or item.get("category")
        severity = item.get("severity")

        if isinstance(label, str) and isinstance(severity, str):
            severities[label.strip().lower()] = severity.strip().lower()

    return severities


def validate_all_cases(cases):
    """Check every case BEFORE running, so we do not fail in the middle."""

    for index, case in enumerate(cases, start=1):
        case_id = get_case_id(case, index)
        text = case.get("contract_text")

        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Missing contract_text in case {case_id}")

        if case.get("language", "en").lower() not in ("ar", "en"):
            raise ValueError(f"Language must be 'ar' or 'en' in case {case_id}")

        get_expected_risks(case)


def select_cases(cases, args):
    """
    python benchmark.py            -> all cases
    python benchmark.py 3          -> first 3 cases
    python benchmark.py V3_AR05    -> only this case
    python benchmark.py A B C      -> only these cases
    """

    if not args:
        return cases

    if len(args) == 1 and args[0].isdigit():
        return cases[: int(args[0])]

    wanted = {arg.strip().lower() for arg in args}
    selected = [
        case
        for index, case in enumerate(cases, start=1)
        if get_case_id(case, index).lower() in wanted
    ]

    found = {get_case_id(case).lower() for case in selected}
    missing = wanted - found

    if missing:
        raise ValueError(f"Case IDs not found in file: {sorted(missing)}")

    return selected


# --------------------------------------------------
# Extract predictions
# --------------------------------------------------


def get_predicted_risks(analysis):
    labels = []

    for risk in analysis.risks:
        if risk.risk_type not in RISK_TYPES:
            raise ValueError(f"Unexpected risk_type: {risk.risk_type}")

        labels.append(risk.risk_type)

    return sorted(set(labels))


def get_predicted_severities(analysis):
    """One severity per category. Keep the highest if repeated."""

    priority = {"low": 1, "medium": 2, "high": 3}
    severities = {}

    for risk in analysis.risks:
        current = severities.get(risk.risk_type)

        if current is None or priority[risk.severity] > priority[current]:
            severities[risk.risk_type] = risk.severity

    return severities


# --------------------------------------------------
# Evaluate one case
# --------------------------------------------------


def evaluate_case(case, index=0):
    case_id = get_case_id(case, index)
    language = case.get("language", "en").lower()
    contract_text = case.get("contract_text", "")
    legal_context = case.get("legal_context", "")

    if not isinstance(legal_context, str):
        legal_context = ""

    expected_risks = get_expected_risks(case)
    expected_severities = get_expected_severities(case)
    expected_verdict = "risky" if expected_risks else "safe"

    result = {
        "model": RUN_MODEL,
        "case_id": case_id,
        "language": language,
        "success": False,
        "valid_json": False,
        "valid_schema": False,
        "recovered_by_retry": False,
        "first_try_error": "",
        "error": "",
        "latency_seconds": None,
        "expected_risks": expected_risks,
        "predicted_risks": [],
        "predicted_severities": {},
        "expected_verdict": expected_verdict,
        "predicted_verdict": "",
        "verdict_correct": False,
        "tp": 0,
        "fp": 0,
        "fn": len(expected_risks),
        "severity_correct": 0,
        "severity_total": 0,
        "raw_response": None,
    }

    start_time = time.perf_counter()

    try:
        prompt = build_risk_prompt(
            contract_text=contract_text,
            legal_context=legal_context,
            language=language,
        )

        raw_response = call_model(prompt)
        result["success"] = True
        result["raw_response"] = raw_response

        # Strict check on raw text (no cleaning).
        try:
            json.loads(raw_response)
            result["valid_json"] = True
        except json.JSONDecodeError:
            result["valid_json"] = False

        # valid_schema measures FIRST try only.
        try:
            analysis = parse_analysis_response(raw_response)
            result["valid_schema"] = True
        except ValueError as first_error:
            result["first_try_error"] = str(first_error)[:400]

            # Same behavior as the real app: retry with the error message.
            analysis = analyze_contract(
                contract_text=contract_text,
                legal_context=legal_context,
                generate_fn=call_model,
                language=language,
                max_retries=MAX_RETRIES,
            )
            result["recovered_by_retry"] = True

        predicted_risks = get_predicted_risks(analysis)
        predicted_severities = get_predicted_severities(analysis)

        expected_set = set(expected_risks)
        predicted_set = set(predicted_risks)

        result["predicted_risks"] = predicted_risks
        result["predicted_severities"] = predicted_severities
        result["predicted_verdict"] = analysis.verdict
        result["verdict_correct"] = analysis.verdict == expected_verdict

        result["tp"] = len(expected_set & predicted_set)
        result["fp"] = len(predicted_set - expected_set)
        result["fn"] = len(expected_set - predicted_set)

        for label in expected_set & predicted_set:
            expected_sev = expected_severities.get(label)
            predicted_sev = predicted_severities.get(label)

            if expected_sev in SEVERITIES and predicted_sev in SEVERITIES:
                result["severity_total"] += 1

                if expected_sev == predicted_sev:
                    result["severity_correct"] += 1

    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"

    finally:
        result["latency_seconds"] = round(time.perf_counter() - start_time, 3)

    return result


# --------------------------------------------------
# Calculate metrics
# --------------------------------------------------


def has_final_analysis(row):
    """True if we got a valid analysis (first try or after retry)."""

    return row["valid_schema"] or (
        row["recovered_by_retry"] and not row["error"]
    )


def calculate_summary(results):
    total = len(results)

    if total == 0:
        return {"model": RUN_MODEL, "total_cases": 0}

    successful = sum(r["success"] for r in results)
    valid_json = sum(r["valid_json"] for r in results)
    valid_schema = sum(r["valid_schema"] for r in results)
    recovered = sum(
        r["recovered_by_retry"] and not r["error"] for r in results
    )

    tp = sum(r["tp"] for r in results)
    fp = sum(r["fp"] for r in results)
    fn = sum(r["fn"] for r in results)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    latencies = sorted(
        r["latency_seconds"]
        for r in results
        if r["latency_seconds"] is not None and r["success"]
    )

    if latencies:
        mean_latency = sum(latencies) / len(latencies)
        p95_index = max(0, int(0.95 * len(latencies) + 0.999999) - 1)
        p95_latency = latencies[min(p95_index, len(latencies) - 1)]
    else:
        mean_latency = None
        p95_latency = None

    # Clean cases = no risk expected.
    clean_cases = [r for r in results if not r["expected_risks"]]
    clean_with_alarm = [
        r for r in clean_cases
        if has_final_analysis(r) and r["predicted_risks"]
    ]
    clean_marked_safe = [
        r for r in clean_cases if r["predicted_verdict"] == "safe"
    ]

    false_alarm_rate = (
        len(clean_with_alarm) / len(clean_cases) if clean_cases else None
    )

    severity_correct = sum(r["severity_correct"] for r in results)
    severity_total = sum(r["severity_total"] for r in results)

    verdict_correct = sum(r["verdict_correct"] for r in results)

    return {
        "model": RUN_MODEL,
        "total_cases": total,
        "request_success_rate": successful / total,
        "valid_json_rate": valid_json / total,
        "valid_schema_rate_first_try": valid_schema / total,
        "recovered_by_retry": recovered,
        "valid_schema_rate_after_retry": (valid_schema + recovered) / total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "clean_case_count": len(clean_cases),
        "clean_cases_with_alarm": len(clean_with_alarm),
        "false_alarm_rate": false_alarm_rate,
        "clean_cases_marked_safe": len(clean_marked_safe),
        "verdict_accuracy": verdict_correct / total,
        "severity_accuracy": (
            severity_correct / severity_total if severity_total else None
        ),
        "severity_correct": severity_correct,
        "severity_total": severity_total,
        "mean_latency_seconds": mean_latency,
        "p95_latency_seconds": p95_latency,
        "error_count": sum(bool(r["error"]) for r in results),
    }


# --------------------------------------------------
# Save results
# --------------------------------------------------


def save_results(results, summary, timestamp):
    """Writes the SAME files during one run (no pile-up of files)."""

    prefix = f"{RUN_MODEL}_{timestamp}"

    jsonl_path = OUTPUT_DIR / f"{prefix}_results.jsonl"
    csv_path = OUTPUT_DIR / f"{prefix}_details.csv"
    summary_path = OUTPUT_DIR / f"{prefix}_summary.json"

    with jsonl_path.open("w", encoding="utf-8") as file:
        for row in results:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    csv_columns = [
        "model",
        "case_id",
        "language",
        "success",
        "valid_json",
        "valid_schema",
        "recovered_by_retry",
        "first_try_error",
        "error",
        "latency_seconds",
        "expected_risks",
        "predicted_risks",
        "predicted_severities",
        "expected_verdict",
        "predicted_verdict",
        "verdict_correct",
        "tp",
        "fp",
        "fn",
        "severity_correct",
        "severity_total",
    ]

    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=csv_columns,
            extrasaction="ignore",
        )
        writer.writeheader()

        for row in results:
            csv_row = dict(row)

            for key in (
                "expected_risks",
                "predicted_risks",
                "predicted_severities",
            ):
                csv_row[key] = json.dumps(
                    csv_row[key],
                    ensure_ascii=False,
                )

            writer.writerow(csv_row)

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4, ensure_ascii=False)

    return jsonl_path, csv_path, summary_path


# --------------------------------------------------
# Main
# --------------------------------------------------


def main():
    if RUN_MODEL not in MODEL_CONFIG:
        raise ValueError(
            f"Invalid RUN_MODEL: {RUN_MODEL}. Choose one of {list(MODEL_CONFIG)}"
        )

    cases = load_cases()
    validate_all_cases(cases)
    cases = select_cases(cases, sys.argv[1:])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = []

    print(f"Model: {RUN_MODEL}")
    print(f"Cases: {len(cases)}")
    print(f"Cases file: {CASES_PATH}")

    if RUN_MODEL != "Nemotron":
        print(f"API URL: {get_generate_url()}")

    print("-" * 50)

    for index, case in enumerate(cases, start=1):
        case_id = get_case_id(case, index)
        print(f"[{index}/{len(cases)}] Running {case_id}...")

        result = evaluate_case(case, index)
        results.append(result)

        print(
            f"  Success={result['success']} | "
            f"Schema={result['valid_schema']} | "
            f"Retry={result['recovered_by_retry']} | "
            f"Verdict={result['predicted_verdict'] or '-'} "
            f"(expected {result['expected_verdict']}) | "
            f"Latency={result['latency_seconds']}s | "
            f"TP={result['tp']} FP={result['fp']} FN={result['fn']}"
        )

        if result["first_try_error"]:
            print(f"  First try error: {result['first_try_error']}")

        if result["error"]:
            print(f"  Error: {result['error']}")

        # Stop early if the first request cannot reach the model.
        if index == 1 and not result["success"]:
            print("\nFirst request failed. Fix the connection first, then run again.")
            print("Check: server running? ngrok URL new? .env updated?")
            save_results(results, calculate_summary(results), timestamp)
            return

        save_results(results, calculate_summary(results), timestamp)

    summary = calculate_summary(results)
    jsonl_path, csv_path, summary_path = save_results(
        results,
        summary,
        timestamp,
    )

    print("\n" + "=" * 50)
    print("BENCHMARK SUMMARY")
    print("=" * 50)

    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nSaved files:")
    print(jsonl_path)
    print(csv_path)
    print(summary_path)


if __name__ == "__main__":
    main()
