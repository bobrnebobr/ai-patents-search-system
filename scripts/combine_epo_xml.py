#!/usr/bin/env python3
"""Combine EPO OPS XML pages into one valid exchange-documents XML file."""

from __future__ import annotations

import argparse
from pathlib import Path
from xml.etree import ElementTree

EXCHANGE_NAMESPACE = "http://www.epo.org/exchange"
OPS_NAMESPACE = "http://ops.epo.org"
XLINK_NAMESPACE = "http://www.w3.org/1999/xlink"

ElementTree.register_namespace("", EXCHANGE_NAMESPACE)
ElementTree.register_namespace("ops", OPS_NAMESPACE)
ElementTree.register_namespace("xlink", XLINK_NAMESPACE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine raw EPO OPS pages into one XML file.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw/epo"),
        help="Directory containing raw XML pages",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/exports/epo_patents.xml"),
        help="Combined XML file",
    )
    return parser.parse_args()


def local_name(tag: str) -> str:
    return tag.rsplit("}", maxsplit=1)[-1]


def document_key(document: ElementTree.Element) -> tuple[str, str, str]:
    return (
        document.attrib.get("country", ""),
        document.attrib.get("doc-number", ""),
        document.attrib.get("kind", ""),
    )


def combine_xml(input_dir: Path, output: Path) -> tuple[int, int, int]:
    xml_paths = sorted(input_dir.rglob("*.xml"))
    if not xml_paths:
        raise SystemExit(f"No XML files found in {input_dir}")

    seen: set[tuple[str, str, str]] = set()
    records_read = 0
    duplicates_skipped = 0

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_suffix(".xml.tmp")
    with temporary_output.open("wb") as destination:
        destination.write(b'<?xml version="1.0" encoding="UTF-8"?>\n')
        destination.write(
            (
                f'<exchange-documents xmlns="{EXCHANGE_NAMESPACE}" '
                f'xmlns:ops="{OPS_NAMESPACE}" xmlns:xlink="{XLINK_NAMESPACE}">\n'
            ).encode()
        )
        for xml_path in xml_paths:
            page_root = ElementTree.parse(xml_path).getroot()
            for document in page_root.iter():
                if local_name(document.tag) != "exchange-document":
                    continue
                records_read += 1
                key = document_key(document)
                if key in seen:
                    duplicates_skipped += 1
                    continue
                seen.add(key)
                destination.write(ElementTree.tostring(document, encoding="UTF-8"))
                destination.write(b"\n")
        destination.write(b"</exchange-documents>\n")
    temporary_output.replace(output)
    return len(xml_paths), records_read - duplicates_skipped, duplicates_skipped


def main() -> None:
    args = parse_args()
    files, records, duplicates = combine_xml(args.input_dir, args.output)
    print(f"Combined {files} XML pages into {args.output}")
    print(f"Saved {records} records; skipped {duplicates} duplicates")


if __name__ == "__main__":
    main()
