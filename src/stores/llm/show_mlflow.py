
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient


DB_PATH = Path(__file__).resolve().parent / "mlflow.db"

mlflow.set_tracking_uri(f"sqlite:///{DB_PATH.as_posix()}")

client = MlflowClient()
experiment = client.get_experiment_by_name("contract-llm")

print(f"MLflow database: {DB_PATH}")

if experiment is None:
    print("No 'contract-llm' experiment found yet.")
else:
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=20,
    )

    if not runs:
        print("No benchmark runs found yet.")
    else:
        for index, run in enumerate(runs, start=1):
            print("\n" + "=" * 55)
            print(f"Run #{index}")
            print(f"Run ID: {run.info.run_id}")
            print(f"Status: {run.info.status}")

            print("\nParameters:")
            for key, value in run.data.params.items():
                print(f"  {key}: {value}")

            print("\nMetrics:")
            for key, value in sorted(run.data.metrics.items()):
                print(f"  {key}: {value:.4f}")

            print("\nArtifacts:")
            print(f"  {run.info.artifact_uri}")
