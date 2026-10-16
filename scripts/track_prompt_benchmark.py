"""Publish an existing holdout comparison to MLflow without rerunning models."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from scripts.benchmark_relevance import summarize, validate_predictions
from scripts.search_laya_prompts import ROOT, TEST, load_dataset

MODELS = {
    "selected_laya": ("laya-task-then-requirements-holdout", "2", "test_selected.jsonl"),
    "champion_v1": ("bge-m3-mlp-holdout", "1", "test_champion.jsonl"),
}


def validate_result(frame: pd.DataFrame, path: Path, expected: dict) -> dict:
    result = pd.read_json(path, lines=True)
    if not result[frame.columns].equals(frame):
        raise ValueError(f"Prediction references differ from benchmark: {path}")
    validate_predictions(
        result[["prediction", "p_not_relevant", "p_uncertain", "p_relevant"]].rename(
            columns={"prediction": "label"}
        ),
        len(frame),
    )
    actual = summarize(result)
    for key, value in expected["overall"].items():
        if not np.isclose(actual["overall"][key], value, atol=1e-8, rtol=1e-8):
            raise ValueError(f"Stored metric differs from predictions: {key}")
    return actual


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Directory containing the completed report")
    args = parser.parse_args()
    directory = args.directory
    report = json.loads((directory / "report.json").read_text())
    frame, manifest = load_dataset(TEST)
    if report["plan"]["test"]["sha256"] != manifest["sha256"]:
        raise ValueError("Report and holdout dataset differ")
    selected_path = directory / "selected_prompt.json"
    selected = json.loads(selected_path.read_text())
    if selected["name"] != report["selected_prompt"]:
        raise ValueError("Selected prompt differs from report")
    results = {
        key: validate_result(frame, directory / filename, report["test"][key])
        for key, (_, _, filename) in MODELS.items()
    }
    files = [directory / "report.json", selected_path]
    files.extend(directory / entry[2] for entry in MODELS.values())
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    study_sha = hashlib.sha256(
        json.dumps({"dataset": manifest["sha256"], "files": hashes}, sort_keys=True).encode()
    ).hexdigest()
    load_dotenv(ROOT / ".env")
    os.environ["MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD"] = "false"
    os.environ["MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD"] = "false"
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    experiment = mlflow.set_experiment("patent-classifier")
    client = mlflow.MlflowClient()
    existing = client.search_runs(
        [experiment.experiment_id],
        filter_string=(
            f"tags.`benchmark.study_sha256` = '{study_sha}' and attributes.status = 'FINISHED'"
        ),
        max_results=1,
    )
    if existing:
        print("Already tracked:", existing[0].info.run_id)
        return
    aliases = client.get_registered_model("patent-classifier").aliases
    versions = {
        version: client.get_model_version("patent-classifier", version) for version in ["1", "2"]
    }
    receipt = {"children": {}, "benchmark_sha256": manifest["sha256"]}
    with mlflow.start_run(run_name="prompt-search-holdout-64-comparison") as parent:
        receipt["parent_run_id"] = parent.info.run_id
        mlflow.set_tags(
            {
                "benchmark.study_sha256": study_sha,
                "benchmark.sha256": manifest["sha256"],
                "evaluation.split": "holdout",
                "reference_labels": "assistant_authored_not_human_verified",
                "human_reviewed": "false",
            }
        )
        mlflow.log_params(
            {
                "pairs": len(frame),
                "families": frame.group_id.nunique(),
                "selected_prompt": selected["name"],
                "prompt_candidates": len(report["dev"]),
            }
        )
        for filename in ["benchmark.jsonl", "manifest.json", "cases.json"]:
            mlflow.log_artifact(str(TEST / filename), "dataset")
        for filename in ["report.json", "plan.json", "selected_prompt.json", "dev_metrics.json"]:
            mlflow.log_artifact(str(directory / filename), "study")
        mlflow.log_dict(hashes, "study/file_sha256.json")
        for filename in [__file__, str(ROOT / "scripts/search_laya_prompts.py")]:
            mlflow.log_artifact(filename, "reproduction")
        artifact_path = parent.info.artifact_uri.removeprefix("mlflow-artifacts:/")
        source = (
            f"{mlflow.get_tracking_uri().rstrip('/')}/api/2.0/mlflow-artifacts/artifacts/"
            f"{artifact_path}/dataset/benchmark.jsonl"
        )
        dataset = mlflow.data.from_pandas(
            frame, source=source, targets="label", name=manifest["name"]
        )
        mlflow.log_input(dataset, context="evaluation")
        for key, (name, version, filename) in MODELS.items():
            with mlflow.start_run(run_name=name, nested=True) as child:
                receipt["children"][key] = child.info.run_id
                override = key == "selected_laya"
                mlflow.set_tags(
                    {
                        "evaluation.split": "holdout",
                        "benchmark.sha256": manifest["sha256"],
                        "human_reviewed": "false",
                        "model.prompt_override": str(override).lower(),
                    }
                )
                mlflow.log_params(
                    {
                        "base_model_uri": f"models:/patent-classifier/{version}",
                        "base_model_source": versions[version].source,
                        "prompt_configuration": selected["name"] if override else "registered_v1",
                        "decision_rule": selected["decision_rule"]
                        if override
                        else "registered_v1_threshold",
                        "prompt_tuned_on_holdout": False,
                    }
                )
                mlflow.log_input(dataset, context="evaluation")
                summary = results[key]
                mlflow.log_metrics(summary["overall"])
                for language, scores in summary["by_language"].items():
                    mlflow.log_metrics(
                        {f"{language}_{metric}": value for metric, value in scores.items()}
                    )
                for kind, scores in summary["by_case_type"].items():
                    mlflow.log_metrics(
                        {f"{kind}_{metric}": value for metric, value in scores.items()}
                    )
                mlflow.log_dict(summary, "metrics.json")
                mlflow.log_artifact(str(directory / filename), "predictions")
                if override:
                    mlflow.log_artifact(str(selected_path), "configuration")
                    mlflow.log_artifact(str(directory / "test_selected.raw.json"), "predictions")
                print(key, child.info.run_id, summary["overall"], flush=True)
        if client.get_registered_model("patent-classifier").aliases != aliases:
            raise RuntimeError("Registry aliases changed during publication")
        mlflow.log_dict(receipt, "study/receipt.json")
    (directory / "mlflow_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
