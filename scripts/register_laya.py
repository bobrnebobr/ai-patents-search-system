"""Register a pinned, self-contained Laya checkpoint as challenger, preserving champion."""

import argparse
import hashlib
import json
import os
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import mlflow
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from mlflow import MlflowClient
from mlflow.models import infer_signature

from scripts.laya_model import QUESTIONS, LayaClassifier

ROOT = Path(__file__).resolve().parents[1]
REVISION = "7b928d828b7b0e022f929d9bd2e44165aa270148"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="patent-classifier")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    os.environ.setdefault("MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD", "false")
    os.environ.setdefault("MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD", "false")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "1800")
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment("patent-classifier")
    client = MlflowClient()
    champion = client.get_registered_model(args.model_name).aliases.get("champion")
    settings = {
        "repo_id": "convaiinnovations/laya",
        "revision": REVISION,
        "subfolder": "multilingual",
        "runtime": f"laya=={version('laya')}",
        "max_len": 2048,
        "head_max_len": 256,
        "batch_size": 8,
        "questions": QUESTIONS,
        "decision_rule": "argmax",
        "mode": "zero_shot",
        "calibration": "checkpoint defaults; not calibrated on patents",
    }
    settings_sha = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    for model in client.search_model_versions(f"name = '{args.model_name}'"):
        if model.tags.get("laya.settings_sha256") == settings_sha:
            client.set_registered_model_alias(args.model_name, "challenger", model.version)
            print(f"Existing Laya version {model.version} assigned @challenger")
            return
    work = ROOT / "data/models" / f"laya-{REVISION}"
    snapshot_download(
        settings["repo_id"],
        revision=REVISION,
        allow_patterns=["multilingual/*"],
        local_dir=work,
    )
    settings_path = work / "relevance_settings.json"
    settings_path.write_text(json.dumps(settings, indent=2) + "\n")
    model = LayaClassifier()
    model.load_context(
        SimpleNamespace(
            artifacts={
                "checkpoint": str(work / "multilingual"),
                "settings": str(settings_path),
            }
        )
    )
    example = pd.DataFrame(
        [
            {
                "query": "Methods for recycling spent lithium batteries",
                "title": "Recovery of metals from spent batteries",
                "abstract": "A process recovers lithium and nickel from spent batteries.",
            },
            {
                "query": "Переработка отработанных литиевых аккумуляторов",
                "title": "Recovery of metals from spent batteries",
                "abstract": "A process recovers lithium and nickel from spent batteries.",
            },
        ]
    )
    output = model.predict(None, example)
    assert len(output) == len(example) and output["label"].notna().all()
    print("Laya RU/EN smoke predictions:", output.to_dict(orient="records"), flush=True)
    with mlflow.start_run(run_name="laya-multilingual-zero-shot-challenger") as run:
        mlflow.set_tags(
            {
                "model.kind": "laya_zero_shot",
                "model.role": "challenger",
                "model.languages": "en,ru",
                "evaluation.status": "smoke_test_only",
                "benchmark.completed": "false",
            }
        )
        mlflow.log_params(
            {
                "hf_model_id": settings["repo_id"],
                "hf_revision": REVISION,
                "hf_subfolder": "multilingual",
                "max_len": settings["max_len"],
                "head_max_len": settings["head_max_len"],
                "decision_rule": "argmax",
                "input_contract": "query,title,abstract",
                "laya_version": version("laya"),
            }
        )
        mlflow.log_artifact(str(settings_path), "configuration")
        mlflow.log_dict(
            {
                "predictions": output.to_dict(orient="records"),
                "note": "Connectivity/shape smoke test, not a quality benchmark",
            },
            "smoke_test.json",
        )
        info = mlflow.pyfunc.log_model(
            name="laya_model",
            python_model=str(ROOT / "scripts/laya_model.py"),
            artifacts={"checkpoint": str(work / "multilingual"), "settings": str(settings_path)},
            signature=infer_signature(example, output),
            input_example=example,
            pip_requirements=[
                f"{package}=={version(package)}"
                for package in [
                    "mlflow",
                    "laya",
                    "torch",
                    "transformers",
                    "huggingface-hub",
                    "safetensors",
                    "numpy",
                    "pandas",
                ]
            ],
        )
        registered = mlflow.register_model(info.model_uri, args.model_name)
        client.set_model_version_tag(
            args.model_name, registered.version, "model.kind", "laya_zero_shot"
        )
        client.set_model_version_tag(args.model_name, registered.version, "hf_revision", REVISION)
        client.set_model_version_tag(
            args.model_name, registered.version, "laya.settings_sha256", settings_sha
        )
        positive_matches = int((output["label"] == "relevant").sum())
        for key, value in {
            "input_contract": "query,title,abstract",
            "evaluation.status": "requires_benchmark",
            "smoke.positive_matches": f"{positive_matches}/{len(example)}",
        }.items():
            client.set_model_version_tag(args.model_name, registered.version, key, value)
        client.update_model_version(
            args.model_name,
            registered.version,
            description=(
                "Multilingual Laya zero-shot challenger. Requires raw query/title/abstract, "
                "unlike v1 embedding-head input. An adapter is required before swapping aliases "
                "in the existing API. Not calibrated on patents; benchmark required."
            ),
        )
        client.set_registered_model_alias(args.model_name, "challenger", registered.version)
        assert client.get_registered_model(args.model_name).aliases.get("champion") == champion
        print(
            f"Registered {args.model_name} v{registered.version} @challenger; champion={champion}"
        )
        print(f"Run: {run.info.run_id}")


if __name__ == "__main__":
    main()
