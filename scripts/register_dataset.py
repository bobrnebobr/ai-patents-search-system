from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mlflow
import pandas as pd

TABLES = (
    "patents",
    "patent_texts",
    "classifications",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--experiment", default="patent-data-eda")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed/epo_english_v1"),
    )
    parser.add_argument(
        "--raw-source",
        default=("s3://datasets/epo/english/v1/epo_english_3g_20261006.xml"),
    )
    parser.add_argument(
        "--processed-base-uri",
        default="s3://datasets/epo/english/v1/processed",
    )
    parser.add_argument("--dataset-version", default="epo_english_v1")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def main() -> None:
    args = parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)

    tracked_tables: list[dict[str, str | int]] = []

    with mlflow.start_run(run_name=f"{args.dataset_version}-dataset-registration") as run:
        mlflow.set_tags(
            {
                "dataset.name": "epo_english",
                "dataset.version": args.dataset_version,
                "dataset.raw_source": args.raw_source,
                "pipeline.step": "dataset_tracking",
            }
        )

        for table_name in TABLES:
            parquet_path = args.data_dir / f"{table_name}.parquet"

            if not parquet_path.exists():
                raise FileNotFoundError(f"Не найден файл: {parquet_path}")

            table_source = f"{args.processed_base_uri}/{table_name}.parquet"
            dataframe = pd.read_parquet(parquet_path)

            full_sha256 = sha256_file(parquet_path)
            mlflow_digest = full_sha256[:32]

            dataset = mlflow.data.from_pandas(
                dataframe,
                name=f"epo_english_{table_name}_{args.dataset_version}",
                source=table_source,
                digest=mlflow_digest,
            )

            mlflow.log_input(
                dataset,
                context="eda",
                tags={
                    "table": table_name,
                    "dataset_version": args.dataset_version,
                    "raw_xml_source": args.raw_source,
                    "derived_by": "scripts/build_dataset.py",
                },
            )

            mlflow.log_metrics(
                {
                    f"dataset.{table_name}.rows": len(dataframe),
                    f"dataset.{table_name}.columns": len(dataframe.columns),
                    f"dataset.{table_name}.size_bytes": parquet_path.stat().st_size,
                }
            )

            tracked_tables.append(
                {
                    "table": table_name,
                    "source": table_source,
                    "rows": len(dataframe),
                    "columns": len(dataframe.columns),
                    "mlflow_digest": mlflow_digest,
                    "sha256": full_sha256,
                }
            )

        mlflow.log_dict(
            {
                "dataset_name": "epo_english",
                "dataset_version": args.dataset_version,
                "raw_source": args.raw_source,
                "transformation": "scripts/build_dataset.py",
                "tables": tracked_tables,
            },
            "dataset_lineage.json",
        )

        print(f"Run ID: {run.info.run_id}")
        print(json.dumps(tracked_tables, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
