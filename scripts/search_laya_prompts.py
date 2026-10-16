"""Select a prompt on the old diagnostic set, then evaluate once on unseen families."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from scripts.benchmark_relevance import LABELS, predict_champion, summarize, validate_predictions

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "research/laya_prompt_search/prompts.json"
DEV = ROOT / "research/relevance_benchmark_v1"
TEST = ROOT / "research/relevance_prompt_holdout_v1"


def load_dataset(directory: Path) -> tuple[pd.DataFrame, dict]:
    path = directory / "benchmark.jsonl"
    manifest = json.loads((directory / "manifest.json").read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError(f"Dataset changed: {path}")
    return pd.read_json(path, lines=True), manifest


def make_states(frame: pd.DataFrame, state_format: str) -> list:
    states = []
    for row in frame.to_dict(orient="records"):
        patent = f"{row['title']}. {row['abstract']}"
        if state_format == "fields":
            state = {key: row[key] for key in ["query", "title", "abstract"]}
        elif state_format == "plain":
            state = f"Search request: {row['query']}\nPatent: {patent}"
        elif state_format == "nli":
            state = {
                "premise": patent,
                "hypothesis": f"The patent provides a solution to this request: {row['query']}",
            }
        else:
            raise ValueError(f"Unknown state format: {state_format}")
        states.append(state)
    return states


def decode(answers: dict, decision_rule: str = "choice") -> dict:
    if decision_rule == "task_requirements_product":
        task = answers["task"]["probabilities"]["yes"]
        conditions = answers["requirements"]["probabilities"]
        # An explicitly defined heuristic combination, not calibrated joint probabilities.
        probabilities = {
            "not_relevant": (1 - task) + task * conditions["A"],
            "uncertain": task * conditions["B"],
            "relevant": task * conditions["C"],
        }
        label = max(probabilities, key=probabilities.get)
    else:
        answer = answers["relevance"]["probabilities"]
        probabilities = dict(zip(LABELS, [answer["A"], answer["B"], answer["C"]], strict=True))
        label = dict(zip(["A", "B", "C"], LABELS, strict=True))[answers["relevance"]["choice"]]
    return {
        **{f"p_{label}": float(value) for label, value in probabilities.items()},
        "label": label,
    }


def evaluate(agent: object, frame: pd.DataFrame, prompt: dict, path: Path) -> dict:
    states = make_states(frame, prompt["state_format"])
    predictions, raw = [], []
    for start in tqdm(range(0, len(states), 4), desc=prompt["name"]):
        results = agent.predict_batch(
            states[start : start + 4],
            prompt["questions"],
            batch_size=4,
            max_len=2048,
            head_max_len=256,
        )
        for result in results:
            if result["usage"].get("truncated") or result["usage"].get("options"):
                raise ValueError("A prompt or input was truncated or options collapsed")
            raw.append(result)
            predictions.append(decode(result["answers"], prompt.get("decision_rule", "choice")))
    predicted = pd.DataFrame(predictions)
    validate_predictions(predicted, len(frame))
    result = pd.concat([frame, predicted.rename(columns={"label": "prediction"})], axis=1)
    result.to_json(path, orient="records", lines=True, force_ascii=False)
    path.with_suffix(".raw.json").write_text(json.dumps(raw, ensure_ascii=False) + "\n")
    return summarize(result)


def select_prompt(summaries: dict) -> str:
    # Frozen selection rule: 3-class macro-F1; ties by relevant AP, then prompt name.
    return min(
        summaries,
        key=lambda name: (
            -summaries[name]["overall"]["macro_f1"],
            -summaries[name]["overall"]["relevant_average_precision"],
            name,
        ),
    )


def main() -> None:
    import laya
    import torch

    torch.set_num_threads(4)
    dev, dev_manifest = load_dataset(DEV)
    test, test_manifest = load_dataset(TEST)
    if set(dev["group_id"]) & set(test["group_id"]):
        raise ValueError("Patent families overlap between dev and test")
    prompts = json.loads(PROMPTS.read_text())
    output = (
        ROOT
        / "data/evaluation"
        / ("laya_prompt_search_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    )
    output.mkdir(parents=True, exist_ok=False)
    plan = {
        "dev": dev_manifest,
        "test": test_manifest,
        "prompts_sha256": hashlib.sha256(PROMPTS.read_bytes()).hexdigest(),
        "prompts": prompts,
        "selection_rule": "max dev macro-F1; tie by relevant AP, then lexical name",
        "checkpoint": "registered patent-classifier v2 multilingual weights; unchanged",
        "max_len": 2048,
        "head_max_len": 256,
        "limitations": [
            "Dev has already been inspected; only test families are fresh for selection.",
            "All labels assistant-authored, not independently human verified.",
            "NLI-inspired request matching, not a standard NLI dataset or verbatim author prompt.",
            "The two-question product is a heuristic, not calibrated joint probabilities.",
        ],
        "sources": [
            "https://nandhakishorm.github.io/laya/questions-and-answers/",
            "https://github.com/NandhaKishorM/laya/tree/4ac226d5ee88ed6df8db288a05c766d7a34ee682",
        ],
    }
    (output / "plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n")
    agent = laya.load(
        str(ROOT / "data/models/registry-laya-v2/artifacts/multilingual"),
        device="cpu",
        backend="eager",
    )
    scores = {}
    for prompt in prompts:
        scores[prompt["name"]] = evaluate(
            agent, dev, prompt, output / f"dev_{prompt['name']}.jsonl"
        )
        (output / "dev_metrics.json").write_text(json.dumps(scores, indent=2) + "\n")
        print(prompt["name"], scores[prompt["name"]]["overall"], flush=True)
    winner = select_prompt(scores)
    selected = next(prompt for prompt in prompts if prompt["name"] == winner)
    # Commit selection to disk before predicting any holdout labels. No re-selection on test.
    (output / "selected_prompt.json").write_text(
        json.dumps(selected, indent=2, ensure_ascii=False) + "\n"
    )
    print("SELECTED ON DEV:", winner, flush=True)
    test_scores = evaluate(agent, test, selected, output / "test_selected.jsonl")
    original = evaluate(agent, test, prompts[0], output / "test_original_v2.jsonl")
    head = predict_champion(test, "http://127.0.0.1:8080", 4)
    validate_predictions(head, len(test))
    head_frame = pd.concat([test, head.rename(columns={"label": "prediction"})], axis=1)
    head_frame.to_json(
        output / "test_champion.jsonl", orient="records", lines=True, force_ascii=False
    )
    report = {
        "selected_prompt": winner,
        "dev": scores,
        "test": {
            "selected_laya": test_scores,
            "original_laya": original,
            "champion_v1": summarize(head_frame),
        },
        "plan": plan,
    }
    baseline = test.copy()
    baseline["prediction"], baseline["p_relevant"] = "not_relevant", 0.0
    report["test"]["constant_not_relevant"] = summarize(baseline)
    (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    for name, summary in report["test"].items():
        print("HOLDOUT", name, json.dumps(summary["overall"]), flush=True)
    print("REPORT:", output / "report.json", flush=True)


if __name__ == "__main__":
    main()
