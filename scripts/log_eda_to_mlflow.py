from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mlflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--experiment", default="patent-data-eda")
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports/eda/epo_v1"),
    )
    parser.add_argument(
        "--source-uri",
        default="s3://datasets/epo_patents_1g.xml",
    )
    parser.add_argument("--dataset-version", default="epo_v1")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    with path.open(encoding="utf-8") as file:
        return json.load(file)


def numeric_metrics(data: dict[str, Any], prefix: str = "") -> dict[str, float]:
    result: dict[str, float] = {}

    for key, value in data.items():
        metric_name = f"{prefix}.{key}" if prefix else key

        if isinstance(value, bool):
            continue

        if isinstance(value, (int, float)):
            result[metric_name] = float(value)

        elif isinstance(value, dict):
            result.update(numeric_metrics(value, metric_name))

    return result


def main() -> None:
    args = parse_args()

    basic_summary = read_json(args.reports_dir / "summary.json")
    advanced_summary = read_json(args.reports_dir / "advanced" / "advanced_summary.json")
    duplicate_summary = read_json(
        args.reports_dir / "advanced" / "duplicate_family_summary.json"
    )

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run(run_name=f"{args.dataset_version}-eda"):
        mlflow.set_tags(
            {
                "dataset.name": "epo_patents",
                "dataset.version": args.dataset_version,
                "dataset.source": args.source_uri,
                "dataset.format": "XML -> Parquet",
                "labels.available": "IPCR",
                "run.type": "EDA",
            }
        )

        mlflow.log_params(
            {
                "raw_format": "XML",
                "processed_tables": "patents, patent_texts, classifications",
                "analysis_engine": "DuckDB",
            }
        )

        metrics = {}
        metrics.update(numeric_metrics(basic_summary, "data"))
        metrics.update(numeric_metrics(advanced_summary, "advanced"))
        metrics.update(numeric_metrics(duplicate_summary, "duplicates"))
        mlflow.log_metrics(metrics)

        dataset_manifest = {
            "name": "epo_patents",
            "version": args.dataset_version,
            "source": args.source_uri,
            "processed_location": "data/processed/epo_v1",
            "tables": [
                "patents.parquet",
                "patent_texts.parquet",
                "classifications.parquet",
            ],
            "label_system": "IPCR",
        }
        mlflow.log_dict(dataset_manifest, "dataset_manifest.json")

        # Загрузит все PNG, JSON и CSV внутри reports/eda/epo_v1.
        mlflow.log_artifacts(str(args.reports_dir), artifact_path="eda")

        print(f"Run ID: {mlflow.active_run().info.run_id}")
        print(f"Artifacts: {mlflow.get_artifact_uri()}")


if __name__ == "__main__":
    main()