"""Freeze an assistant-authored diagnostic benchmark using real corpus abstracts."""

import argparse
import hashlib
import json
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "research/relevance_benchmark_v1"


def build(
    cases_path: Path = DESTINATION / "cases.json",
    destination: Path = DESTINATION,
    name: str = "patent-relevance-diagnostic-v1",
) -> None:
    cases = json.loads(cases_path.read_text())
    source = ROOT / "data/processed/epo_english_v1/patents_search.parquet"
    candidates = ROOT / "data/annotation/patent_relevance_v1/pairs.parquet"
    patents = (
        pl.scan_parquet(source)
        .filter(pl.col("publication_id").is_in([case["publication_id"] for case in cases]))
        .collect()
    )
    seen = pl.scan_parquet(candidates).select("family_id").unique().collect()
    if patents.join(seen, on="family_id", how="inner").height:
        raise ValueError("A benchmark family occurs in the v1 training-candidate pool")
    lookup = {row["publication_id"]: row for row in patents.to_dicts()}
    rows = []
    for index, case in enumerate(cases):
        patent = lookup[case["publication_id"]]
        if case["evidence"] not in patent["abstract"]:
            raise ValueError(f"Evidence does not occur in {case['publication_id']}")
        other = cases[(index + 1) % len(cases)]
        for kind in ["relevant", "hard_negative", "uncertain", "easy_negative"]:
            queries = other["relevant"] if kind == "easy_negative" else case[kind]
            label = "not_relevant" if "negative" in kind else kind
            reason = {
                "relevant": f"Directly supported by: {case['evidence']}",
                "hard_negative": case["negative_reason"],
                "uncertain": case["uncertain_reason"],
                "easy_negative": f"Different technical task: {other['topic']}.",
            }[kind]
            for language, query in zip(["en", "ru"], queries, strict=True):
                rows.append(
                    {
                        "pair_id": f"{patent['publication_id']}.{kind}.{language}",
                        "group_id": patent["family_id"],
                        "translation_group_id": f"{patent['publication_id']}.{kind}",
                        "publication_id": patent["publication_id"],
                        "topic": case["topic"],
                        "query": query,
                        "language": language,
                        "title": patent["title"],
                        "abstract": patent["abstract"],
                        "label": label,
                        "case_type": kind,
                        "reference_reason": reason,
                        "reference_author": "assistant_authored_not_human_verified",
                    }
                )
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "benchmark.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    manifest = {
        "name": name,
        "pairs": len(rows),
        "independent_patent_families": len(cases),
        "languages": ["en", "ru"],
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "cases_sha256": hashlib.sha256(cases_path.read_bytes()).hexdigest(),
        "label_source": "assistant-authored; no independent human review",
        "source": "s3://datasets/epo/english/v1/processed/patents_search.parquet",
        "v1_candidate_family_overlap": 0,
        "leakage_check": str(candidates.relative_to(ROOT)),
        "limitations": [
            "Not a random retrieval sample; constructed queries and balanced case types.",
            "Translations and cases within a family are correlated, "
            f"not {len(rows)} independent trials.",
            "Only title and abstract are evidence; absent conditions are uncertain, not negative.",
            "No guarantee against foundation-model pretraining contamination.",
            "Do not tune thresholds or prompts on this dataset and report it as held-out.",
        ],
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DESTINATION / "cases.json")
    parser.add_argument("--output", type=Path, default=DESTINATION)
    parser.add_argument("--name", default="patent-relevance-diagnostic-v1")
    args = parser.parse_args()
    build(args.cases, args.output, args.name)
