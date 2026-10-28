
from pathlib import Path
import math

import mlflow


# Keep the tracking database in this file's directory.
DB_PATH = Path(__file__).resolve().parent / "mlflow.db"

mlflow.set_tracking_uri(f"sqlite:///{DB_PATH.as_posix()}")
mlflow.set_experiment("contract-llm")


def log_benchmark_run(
    summary,
    artifact_paths,
    dataset_name,
    prompt_version,
):
    """Log one completed benchmark run, metrics, parameters and files."""

    model_name = summary.get("model", "LLM")

    with mlflow.start_run(run_name=f"{model_name}-benchmark"):

        # Run configuration
        mlflow.log_params({
            "model": str(model_name),
            "dataset": str(dataset_name),
            "total_cases": int(summary.get("total_cases", 0)),
            "prompt_version": str(prompt_version),
            "max_retries": 2,
            "evaluation_type": "contract risk analysis",
        })

        # Log numeric metrics only.
        metrics = {}

        for key, value in summary.items():
            if key in {"model", "total_cases"}:
                continue

            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if math.isfinite(value):
                    metrics[key] = value

        if metrics:
            mlflow.log_metrics(metrics)

        # Save benchmark output files as MLflow artifacts.
        for path in artifact_paths:
            if path is None:
                continue

            file_path = Path(path)

            if file_path.exists() and file_path.is_file():
                mlflow.log_artifact(
                    str(file_path),
                    artifact_path="benchmark_results",
                )
            else:
                print(f"Artifact not found, skipped: {file_path}")

        print(f"MLflow run ID: {mlflow.active_run().info.run_id}")

    print(f"MLflow database: {DB_PATH}")
    print("Benchmark results logged to MLflow.")
