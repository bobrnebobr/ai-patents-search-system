#!/usr/bin/env python3
"""Combine raw EPO OPS XML pages into a deduplicated JSONL dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from sample_epo_ops import parse_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine EPO OPS XML pages into one JSONL file.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw/epo"),
        help="Directory containing raw XML pages",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/exports/epo_patents.jsonl"),
        help="Combined JSONL dataset path",
    )
    return parser.parse_args()


def record_key(record: dict[str, Any]) -> str:
    publication_number = record.get("publication_number")
    if publication_number:
        return str(publication_number)
    return "|".join(
        str(record.get(field) or "")
        for field in ("country", "document_number", "kind", "publication_date")
    )


def increment_languages(counter: Counter[str], values: object) -> None:
    if isinstance(values, dict):
        counter.update(str(language) for language in values)


def update_stats(stats: dict[str, Any], record: dict[str, Any]) -> None:
    stats["countries"][str(record.get("country") or "unknown")] += 1
    stats["kinds"][str(record.get("kind") or "unknown")] += 1
    increment_languages(stats["title_languages"], record.get("titles"))
    increment_languages(stats["abstract_languages"], record.get("abstracts"))
    stats["with_title"] += bool(record.get("titles"))
    stats["with_abstract"] += bool(record.get("abstracts"))
    stats["with_cpc"] += bool(record.get("cpc_codes"))
    stats["with_ipc"] += bool(record.get("ipc_codes"))
    stats["with_family_id"] += bool(record.get("family_id"))


def serializable_stats(stats: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in stats.items():
        if isinstance(value, Counter):
            result[name] = dict(value.most_common())
        else:
            result[name] = value
    return result


def build_dataset(input_dir: Path, output: Path) -> dict[str, Any]:
    xml_paths = sorted(input_dir.rglob("*.xml"))
    if not xml_paths:
        raise SystemExit(f"No XML files found in {input_dir}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_suffix(output.suffix + ".tmp")
    seen: set[str] = set()
    stats: dict[str, Any] = {
        "xml_files": len(xml_paths),
        "records_read": 0,
        "records_written": 0,
        "duplicates_skipped": 0,
        "with_title": 0,
        "with_abstract": 0,
        "with_cpc": 0,
        "with_ipc": 0,
        "with_family_id": 0,
        "countries": Counter(),
        "kinds": Counter(),
        "title_languages": Counter(),
        "abstract_languages": Counter(),
    }

    with temporary_output.open("w", encoding="utf-8") as stream:
        for xml_path in xml_paths:
            for record in parse_records(xml_path.read_bytes()):
                stats["records_read"] += 1
                key = record_key(record)
                if key in seen:
                    stats["duplicates_skipped"] += 1
                    continue
                seen.add(key)
                record["source_file"] = xml_path.relative_to(input_dir).as_posix()
                stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                stats["records_written"] += 1
                update_stats(stats, record)

    temporary_output.replace(output)
    result = serializable_stats(stats)
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    args = parse_args()
    stats = build_dataset(args.input_dir, args.output)
    print(f"Saved {stats['records_written']} records to {args.output}")
    print(f"Skipped {stats['duplicates_skipped']} duplicate publications")
    print(f"Summary: {args.output.with_suffix('.summary.json')}")


if __name__ == "__main__":
    main()
