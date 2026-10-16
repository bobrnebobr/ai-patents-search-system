"""Compare pinned classifier versions on the frozen bilingual diagnostic benchmark."""

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import httpx
import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "research/relevance_benchmark_v1"
LABELS = ["not_relevant", "uncertain", "relevant"]


def metrics(frame: pd.DataFrame) -> dict:
    true, predicted = frame["label"], frame["prediction"]
    relevant = (true == "relevant").astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        relevant, predicted == "relevant", average="binary", zero_division=0
    )
    return {
        "n": len(frame),
        "accuracy": float(accuracy_score(true, predicted)),
        "macro_f1": float(f1_score(true, predicted, labels=LABELS, average="macro")),
        "relevant_precision": float(precision),
        "relevant_recall": float(recall),
        "relevant_f1": float(f1),
        "relevant_average_precision": float(average_precision_score(relevant, frame["p_relevant"])),
    }


def summarize(frame: pd.DataFrame) -> dict:
    return {
        "overall": metrics(frame),
        "by_language": {language: metrics(part) for language, part in frame.groupby("language")},
        "by_case_type": {
            kind: {
                "n": len(part),
                "accuracy": float(accuracy_score(part["label"], part["prediction"])),
                "predicted_relevant_rate": float((part["prediction"] == "relevant").mean()),
            }
            for kind, part in frame.groupby("case_type")
        },
        "labels": LABELS,
        "confusion_matrix": confusion_matrix(
            frame["label"], frame["prediction"], labels=LABELS
        ).tolist(),
        "classification_report": classification_report(
            frame["label"], frame["prediction"], labels=LABELS, output_dict=True, zero_division=0
        ),
        "prediction_counts": frame["prediction"].value_counts().to_dict(),
    }


def infer(client: httpx.Client, model: str, inputs: list, pandas: bool = False) -> dict:
    payload = {"inputs": inputs}
    if pandas:
        payload["parameters"] = {"content_type": "pd"}
    response = client.post(f"/v2/models/{model}/infer", json=payload)
    response.raise_for_status()
    return response.json()


def predict_champion(frame: pd.DataFrame, url: str, batch_size: int) -> pd.DataFrame:
    patent_text = frame["title"] + ". " + frame["abstract"]
    texts = list(dict.fromkeys([*frame["query"], *patent_text]))
    cache = {}
    results = []
    with httpx.Client(base_url=url, timeout=300, trust_env=False) as client:
        for name in ["patent-bi-encoder", "patent-classifier"]:
            client.get(f"/v2/models/{name}/ready").raise_for_status()
        for start in tqdm(range(0, len(texts), batch_size), desc="Encoder batches"):
            batch = texts[start : start + batch_size]
            response = infer(
                client,
                "patent-bi-encoder",
                [
                    {
                        "name": "text",
                        "shape": [len(batch)],
                        "datatype": "BYTES",
                        "data": batch,
                        "parameters": {"content_type": "str"},
                    }
                ],
                pandas=True,
            )
            output = response["outputs"][0]
            vectors = np.asarray(output["data"], dtype=np.float32).reshape(output["shape"])
            if vectors.shape != (len(batch), 1024) or not np.isfinite(vectors).all():
                raise ValueError("Encoder returned invalid embeddings")
            cache.update(zip(batch, vectors, strict=True))
        for start in tqdm(range(0, len(frame), batch_size), desc="Classifier batches"):
            stop = min(start + batch_size, len(frame))
            vectors = np.stack(
                [
                    np.concatenate([cache[frame.iloc[i]["query"]], cache[patent_text.iloc[i]]])
                    for i in range(start, stop)
                ]
            )
            response = infer(
                client,
                "patent-classifier",
                [
                    {
                        "name": "embeddings",
                        "shape": list(vectors.shape),
                        "datatype": "FP32",
                        "data": vectors.reshape(-1).tolist(),
                        "parameters": {"content_type": "np"},
                    }
                ],
            )
            outputs = {item["name"]: item["data"] for item in response["outputs"]}
            results.extend(pd.DataFrame(outputs).to_dict(orient="records"))
    return pd.DataFrame(results)


def predict_laya(frame: pd.DataFrame, model_path: Path, batch_size: int) -> pd.DataFrame:
    import torch

    torch.set_num_threads(4)
    model = mlflow.pyfunc.load_model(str(model_path))
    outputs = []
    for start in tqdm(range(0, len(frame), batch_size), desc="Laya batches"):
        batch = frame.iloc[start : start + batch_size][["query", "title", "abstract"]]
        outputs.append(model.predict(batch))
    return pd.concat(outputs, ignore_index=True)


def validate_predictions(predictions: pd.DataFrame, expected: int) -> None:
    if len(predictions) != expected or not predictions["label"].isin(LABELS).all():
        raise ValueError("Invalid prediction labels or row count")
    scores = predictions[[f"p_{label}" for label in LABELS]].to_numpy()
    if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
        raise ValueError("Invalid prediction scores")
    if not np.allclose(scores.sum(axis=1), 1, atol=5e-4):
        raise ValueError("Class probabilities do not sum to one")


def run_local(url: str, laya_path: Path, batch_size: int) -> Path:
    """Run without remote tracking or uploads; retain all predictions on disk."""
    path = BENCHMARK / "benchmark.jsonl"
    manifest = json.loads((BENCHMARK / "manifest.json").read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError("Frozen benchmark digest mismatch")
    frame = pd.read_json(path, lines=True)
    output = ROOT / "data/evaluation" / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "benchmark": manifest,
        "model_uris": {
            "champion": "models:/patent-classifier/1",
            "challenger": "models:/patent-classifier/2",
        },
        "execution": {
            "champion": "Existing local MLServer; encoder + deployed head, unchanged threshold",
            "challenger": str(laya_path),
            "remote_registry_checked_at_runtime": False,
            "timing_caveat": "Different runtimes; not a hardware-normalized comparison",
        },
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "models": {},
    }
    for name in ["champion", "challenger"]:
        started = perf_counter()
        predictions = (
            predict_champion(frame, url, batch_size)
            if name == "champion"
            else predict_laya(frame, laya_path, batch_size)
        )
        elapsed = perf_counter() - started
        validate_predictions(predictions, len(frame))
        evaluated = pd.concat([frame, predictions.rename(columns={"label": "prediction"})], axis=1)
        evaluated.to_json(
            output / f"{name}_predictions.jsonl", orient="records", lines=True, force_ascii=False
        )
        summary = summarize(evaluated)
        summary["elapsed_seconds"] = elapsed
        report["models"][name] = summary
        (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(name, json.dumps(summary, ensure_ascii=False), flush=True)
    print(f"Report: {output / 'report.json'}", flush=True)
    return output / "report.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--local-only", action="store_true", help="Do not write to remote MLflow")
    parser.add_argument(
        "--publish-report", type=Path, help="Track saved results without re-inference"
    )
    parser.add_argument("--laya-path", type=Path, default=ROOT / "data/models/registry-laya-v2")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch size must be positive")
    if args.local_only and args.publish_report:
        parser.error("--local-only cannot be combined with --publish-report")
    if args.local_only:
        run_local(args.url, args.laya_path, args.batch_size)
        return
    load_dotenv(ROOT / ".env")
    os.environ.setdefault("MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD", "false")
    os.environ.setdefault("MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD", "false")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "1800")
    client = mlflow.MlflowClient()
    aliases = client.get_registered_model("patent-classifier").aliases
    if aliases.get("champion") != "1" or aliases.get("challenger") != "2":
        raise ValueError("This benchmark is pinned to champion v1 and challenger v2")
    encoder_version = client.get_model_version_by_alias("patent-bi-encoder", "champion")
    path = BENCHMARK / "benchmark.jsonl"
    manifest = json.loads((BENCHMARK / "manifest.json").read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError("Frozen benchmark digest mismatch")
    frame = pd.read_json(path, lines=True)
    if frame["pair_id"].duplicated().any() or frame["label"].isna().any():
        raise ValueError("Invalid benchmark references")
    saved = json.loads(args.publish_report.read_text()) if args.publish_report else None
    if saved and saved["benchmark"]["sha256"] != manifest["sha256"]:
        raise ValueError("Saved results belong to a different benchmark")
    output = (
        args.publish_report.parent
        if args.publish_report
        else ROOT / "data/evaluation" / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    )
    if not saved:
        output.mkdir(parents=True, exist_ok=False)
    provenance = {
        "models": {
            "champion": "models:/patent-classifier/1",
            "challenger": "models:/patent-classifier/2",
        },
        "encoder": f"models:/patent-bi-encoder/{encoder_version.version}",
        "encoder_execution": "existing local MLServer loaded from champion alias",
        "champion_execution": "existing local MLServer; no threshold changes",
        "laya_execution": "local CPU; exact registered v2 artifact; unchanged prompt/settings",
        "benchmark": manifest,
        "packages": {package: version(package) for package in ["mlflow", "laya", "torch"]},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "timing_caveat": "Different runtimes; timings are not a hardware-normalized comparison.",
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    mlflow.set_experiment("patent-classifier")
    summaries = {}
    with mlflow.start_run(run_name="bilingual-diagnostic-benchmark-v1") as parent:
        mlflow.set_tags(
            {
                "evaluation.kind": "assistant_authored_diagnostic",
                "human_reviewed": "false",
                "benchmark.sha256": manifest["sha256"],
            }
        )
        for filename in ["benchmark.jsonl", "manifest.json", "cases.json"]:
            mlflow.log_artifact(str(BENCHMARK / filename), "dataset")
        mlflow.log_artifact(str(output / "provenance.json"))
        mlflow.log_artifact(__file__, "reproduction")
        artifact_path = parent.info.artifact_uri.removeprefix("mlflow-artifacts:/")
        source = (
            f"{mlflow.get_tracking_uri().rstrip('/')}/api/2.0/mlflow-artifacts/artifacts/"
            f"{artifact_path}/dataset/benchmark.jsonl"
        )
        dataset = mlflow.data.from_pandas(
            frame,
            source=source,
            targets="label",
            name=manifest["name"],
        )
        mlflow.log_input(dataset, context="evaluation")
        for name in ["champion", "challenger"]:
            with mlflow.start_run(run_name=f"{name}-diagnostic-v1", nested=True) as child:
                mlflow.log_input(dataset, context="evaluation")
                mlflow.log_params(
                    {
                        "model_uri": provenance["models"][name],
                        "reference_labels": "assistant_authored_not_human_verified",
                        "threshold_or_prompt_tuned_on_benchmark": False,
                    }
                )
                if saved:
                    existing = pd.read_json(output / f"{name}_predictions.jsonl", lines=True)
                    if not existing[frame.columns].equals(frame):
                        raise ValueError("Saved prediction rows do not match benchmark references")
                    predictions = existing[
                        ["p_not_relevant", "p_uncertain", "p_relevant", "prediction"]
                    ].rename(columns={"prediction": "label"})
                    elapsed = saved["models"][name]["elapsed_seconds"]
                else:
                    started = perf_counter()
                    predictions = (
                        predict_champion(frame, args.url, args.batch_size)
                        if name == "champion"
                        else predict_laya(frame, args.laya_path, args.batch_size)
                    )
                    elapsed = perf_counter() - started
                validate_predictions(predictions, len(frame))
                evaluated = pd.concat(
                    [
                        frame.reset_index(drop=True),
                        predictions.rename(columns={"label": "prediction"}),
                    ],
                    axis=1,
                )
                predictions_path = output / f"{name}_predictions.jsonl"
                evaluated.to_json(predictions_path, orient="records", lines=True, force_ascii=False)
                summary = summarize(evaluated)
                summary.update({"elapsed_seconds": elapsed, "run_id": child.info.run_id})
                summaries[name] = summary
                mlflow.log_metrics({key: value for key, value in summary["overall"].items()})
                for language, scores in summary["by_language"].items():
                    mlflow.log_metrics(
                        {f"{language}_{key}": value for key, value in scores.items()}
                    )
                mlflow.log_dict(summary, "metrics.json")
                mlflow.log_artifact(str(predictions_path), "predictions")
                print(name, json.dumps(summary, ensure_ascii=False), flush=True)
        if client.get_registered_model("patent-classifier").aliases != aliases:
            raise RuntimeError("Registry aliases changed during evaluation; check provenance")
        report = {"parent_run_id": parent.info.run_id, "models": summaries, **provenance}
        report["model_uris"] = provenance["models"]
        report["models"] = summaries
        (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        mlflow.log_artifact(str(output / "report.json"))
        print(f"Report: {output / 'report.json'}", flush=True)
        print(f"MLflow run: {parent.info.run_id}", flush=True)


if __name__ == "__main__":
    main()
