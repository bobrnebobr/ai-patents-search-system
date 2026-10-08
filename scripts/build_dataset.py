from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


PATENTS_SCHEMA = pa.schema(
    [
        pa.field("publication_id", pa.string()),
        pa.field("family_id", pa.string()),
        pa.field("publication_authority", pa.string()),
        pa.field("document_number", pa.string()),
        pa.field("document_kind", pa.string()),
        pa.field("publication_date", pa.date32()),
        pa.field("publication_date_raw", pa.string()),
    ]
)

PATENT_TEXTS_SCHEMA = pa.schema(
    [
        pa.field("publication_id", pa.string()),
        pa.field("source_language", pa.string()),
        pa.field("title", pa.string()),
        pa.field("abstract", pa.string()),
    ]
)

CLASSIFICATIONS_SCHEMA = pa.schema(
    [
        pa.field("publication_id", pa.string()),
        pa.field("classification_system", pa.string()),
        pa.field("classification_code", pa.string()),
        pa.field("raw_classification", pa.string()),
        pa.field("sequence", pa.int32()),
        pa.field("classification_value", pa.string()),
        pa.field("generating_office", pa.string()),
    ]
)


class ParquetBatchWriter:
    def __init__(self, path: Path, schema: pa.Schema) -> None:
        self.writer = pq.ParquetWriter(
            path,
            schema=schema,
            compression="zstd",
        )
        self.schema = schema

    def write(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return

        table = pa.Table.from_pylist(rows, schema=self.schema)
        self.writer.write_table(table)

    def close(self) -> None:
        self.writer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Конвертация EPO XML в English-only Parquet dataset."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2_000)
    parser.add_argument("--max-documents", type=int, default=None)
    return parser.parse_args()


def local_name(tag: str) -> str:
    return tag.rsplit("}", maxsplit=1)[-1]


def direct_children(element: ET.Element | None, name: str) -> list[ET.Element]:
    if element is None:
        return []

    return [child for child in element if local_name(child.tag) == name]


def first_child(element: ET.Element | None, name: str) -> ET.Element | None:
    children = direct_children(element, name)
    return children[0] if children else None


def element_text(element: ET.Element | None) -> str | None:
    if element is None:
        return None

    value = " ".join("".join(element.itertext()).split())
    return value or None


def child_text(element: ET.Element | None, name: str) -> str | None:
    return element_text(first_child(element, name))


def get_attribute(element: ET.Element, name: str) -> str | None:
    for attribute_name, value in element.attrib.items():
        if local_name(attribute_name) == name:
            return value

    return None


def find_text_by_language(
    element: ET.Element | None,
    tag_name: str,
    language: str,
) -> str | None:
    for child in direct_children(element, tag_name):
        child_language = (get_attribute(child, "lang") or "").lower()

        if child_language == language:
            return element_text(child)

    return None


def parse_publication_date(value: str | None) -> date | None:
    if value is None or not re.fullmatch(r"\d{8}", value):
        return None

    try:
        return date.fromisoformat(f"{value[:4]}-{value[4:6]}-{value[6:8]}")
    except ValueError:
        return None


def normalize_component(value: str | None) -> str:
    return re.sub(r"\s+", "", value or "")


def normalize_ipcr_code(raw_value: str) -> str:
    compact_value = " ".join(raw_value.split())

    match = re.search(
        r"([A-HY]\d{2}[A-Z])\s*(\d+)\s*/\s*(\d+)",
        compact_value,
    )

    if match is None:
        return compact_value

    subclass, main_group, subgroup = match.groups()
    return f"{subclass}{main_group}/{subgroup.zfill(2)}"


def build_cpci_code(classification: ET.Element) -> str | None:
    section = normalize_component(child_text(classification, "section"))
    class_code = normalize_component(child_text(classification, "class"))
    subclass = normalize_component(child_text(classification, "subclass"))
    main_group = normalize_component(child_text(classification, "main-group"))
    subgroup = normalize_component(child_text(classification, "subgroup"))

    if not all((section, class_code, subclass, main_group, subgroup)):
        return None

    return f"{section}{class_code}{subclass}{main_group}/{subgroup.zfill(2)}"


def parse_publication_data(
    document: ET.Element,
) -> dict[str, str | None]:
    bibliographic_data = first_child(document, "bibliographic-data")
    publication_reference = first_child(
        bibliographic_data,
        "publication-reference",
    )

    document_ids = direct_children(publication_reference, "document-id")

    docdb_id = next(
        (item for item in document_ids if get_attribute(item, "document-id-type") == "docdb"),
        None,
    )
    epodoc_id = next(
        (item for item in document_ids if get_attribute(item, "document-id-type") == "epodoc"),
        None,
    )

    authority = child_text(docdb_id, "country") or get_attribute(document, "country")
    document_number = child_text(docdb_id, "doc-number") or get_attribute(document, "doc-number")
    document_kind = child_text(docdb_id, "kind") or get_attribute(document, "kind")
    publication_date_raw = child_text(docdb_id, "date")

    epodoc_number = child_text(epodoc_id, "doc-number")

    if epodoc_number:
        publication_id = epodoc_number

        if document_kind and not publication_id.endswith(document_kind):
            publication_id = f"{publication_id}{document_kind}"
    elif authority and document_number:
        publication_id = f"{authority}{document_number}{document_kind or ''}"
    else:
        publication_id = None

    return {
        "publication_id": publication_id,
        "family_id": get_attribute(document, "family-id"),
        "publication_authority": authority,
        "document_number": document_number,
        "document_kind": document_kind,
        "publication_date_raw": publication_date_raw,
    }


def extract_ipcr_rows(
    document: ET.Element,
    publication_id: str,
) -> list[dict[str, Any]]:
    bibliographic_data = first_child(document, "bibliographic-data")
    classifications = first_child(
        bibliographic_data,
        "classifications-ipcr",
    )

    rows: list[dict[str, Any]] = []

    for item in direct_children(classifications, "classification-ipcr"):
        raw_value = element_text(first_child(item, "text"))

        if raw_value is None:
            continue

        sequence_raw = get_attribute(item, "sequence")

        rows.append(
            {
                "publication_id": publication_id,
                "classification_system": "IPCR",
                "classification_code": normalize_ipcr_code(raw_value),
                "raw_classification": raw_value,
                "sequence": int(sequence_raw) if sequence_raw else None,
                "classification_value": None,
                "generating_office": None,
            }
        )

    return rows


def extract_cpci_rows(
    document: ET.Element,
    publication_id: str,
) -> list[dict[str, Any]]:
    bibliographic_data = first_child(document, "bibliographic-data")
    classifications = first_child(
        bibliographic_data,
        "patent-classifications",
    )

    rows: list[dict[str, Any]] = []

    for item in direct_children(classifications, "patent-classification"):
        scheme_element = first_child(item, "classification-scheme")
        scheme = get_attribute(scheme_element, "scheme") if scheme_element is not None else None

        if scheme not in {"CPCI", "CPC"}:
            continue

        classification_code = build_cpci_code(item)

        if classification_code is None:
            continue

        sequence_raw = get_attribute(item, "sequence")

        rows.append(
            {
                "publication_id": publication_id,
                "classification_system": scheme,
                "classification_code": classification_code,
                "raw_classification": None,
                "sequence": int(sequence_raw) if sequence_raw else None,
                "classification_value": child_text(
                    item,
                    "classification-value",
                ),
                "generating_office": child_text(
                    item,
                    "generating-office",
                ),
            }
        )

    return rows


def build_summary(
    *,
    input_path: Path,
    batch_size: int,
    documents_seen: int,
    documents_without_id: int,
    duplicates_skipped: int,
    patents_written: int,
    english_text_rows_written: int,
    ipcr_rows_written: int,
    cpci_rows_written: int,
    patents_without_english_text: int,
    patent_null_counts: Counter[str],
) -> dict[str, Any]:
    return {
        "source_file": str(input_path),
        "batch_size": batch_size,
        "documents_seen": documents_seen,
        "documents_without_id": documents_without_id,
        "duplicates_skipped": duplicates_skipped,
        "patents_written": patents_written,
        "english_text_rows_written": english_text_rows_written,
        "ipcr_rows_written": ipcr_rows_written,
        "cpci_rows_written": cpci_rows_written,
        "classification_rows_written": ipcr_rows_written + cpci_rows_written,
        "patents_without_english_text": patents_without_english_text,
        "patent_null_counts": dict(patent_null_counts),
    }


def main() -> None:
    args = parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(f"Не найден XML-файл: {args.input}")

    if args.batch_size < 1:
        raise ValueError("--batch-size должен быть положительным числом")

    if args.max_documents is not None and args.max_documents < 1:
        raise ValueError("--max-documents должен быть положительным числом")

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Папка уже не пуста: {args.output_dir}. "
            "Выбери новую output-папку, чтобы не перезаписать данные."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    patents_writer = ParquetBatchWriter(
        args.output_dir / "patents.parquet",
        PATENTS_SCHEMA,
    )
    texts_writer = ParquetBatchWriter(
        args.output_dir / "patent_texts.parquet",
        PATENT_TEXTS_SCHEMA,
    )
    classifications_writer = ParquetBatchWriter(
        args.output_dir / "classifications.parquet",
        CLASSIFICATIONS_SCHEMA,
    )

    patent_rows: list[dict[str, Any]] = []
    text_rows: list[dict[str, Any]] = []
    classification_rows: list[dict[str, Any]] = []

    seen_publication_ids: set[str] = set()
    patent_null_counts: Counter[str] = Counter()

    documents_seen = 0
    documents_without_id = 0
    duplicates_skipped = 0
    patents_written = 0
    english_text_rows_written = 0
    ipcr_rows_written = 0
    cpci_rows_written = 0
    patents_without_english_text = 0
    batches_written = 0

    def flush() -> None:
        nonlocal batches_written

        if not patent_rows:
            return

        patents_writer.write(patent_rows)
        texts_writer.write(text_rows)
        classifications_writer.write(classification_rows)

        patent_rows.clear()
        text_rows.clear()
        classification_rows.clear()
        batches_written += 1

    root: ET.Element | None = None

    try:
        context = ET.iterparse(args.input, events=("start", "end"))

        for event, document in context:
            if root is None and event == "start":
                root = document

            if event != "end" or local_name(document.tag) != "exchange-document":
                continue

            documents_seen += 1

            publication_data = parse_publication_data(document)
            publication_id = publication_data["publication_id"]

            if publication_id is None:
                documents_without_id += 1
                document.clear()
                continue

            if publication_id in seen_publication_ids:
                duplicates_skipped += 1
                document.clear()
                continue

            seen_publication_ids.add(publication_id)

            publication_date_raw = publication_data["publication_date_raw"]

            patent_row = {
                "publication_id": publication_id,
                "family_id": publication_data["family_id"],
                "publication_authority": publication_data["publication_authority"],
                "document_number": publication_data["document_number"],
                "document_kind": publication_data["document_kind"],
                "publication_date": parse_publication_date(publication_date_raw),
                "publication_date_raw": publication_date_raw,
            }
            patent_rows.append(patent_row)
            patents_written += 1

            for field_name, value in patent_row.items():
                if value is None:
                    patent_null_counts[field_name] += 1

            bibliographic_data = first_child(
                document,
                "bibliographic-data",
            )
            english_title = find_text_by_language(
                bibliographic_data,
                "invention-title",
                "en",
            )
            english_abstract = find_text_by_language(
                document,
                "abstract",
                "en",
            )

            if english_title or english_abstract:
                text_rows.append(
                    {
                        "publication_id": publication_id,
                        "source_language": "en",
                        "title": english_title,
                        "abstract": english_abstract,
                    }
                )
                english_text_rows_written += 1
            else:
                patents_without_english_text += 1

            ipcr_rows = extract_ipcr_rows(document, publication_id)
            cpci_rows = extract_cpci_rows(document, publication_id)

            classification_rows.extend(ipcr_rows)
            classification_rows.extend(cpci_rows)

            ipcr_rows_written += len(ipcr_rows)
            cpci_rows_written += len(cpci_rows)

            if documents_seen % args.batch_size == 0:
                flush()
                print(f"Обработано документов: {documents_seen:,}")

            document.clear()

            if root is not None:
                root.clear()

            if args.max_documents is not None and documents_seen >= args.max_documents:
                break

        flush()
    finally:
        patents_writer.close()
        texts_writer.close()
        classifications_writer.close()

    summary = build_summary(
        input_path=args.input,
        batch_size=args.batch_size,
        documents_seen=documents_seen,
        documents_without_id=documents_without_id,
        duplicates_skipped=duplicates_skipped,
        patents_written=patents_written,
        english_text_rows_written=english_text_rows_written,
        ipcr_rows_written=ipcr_rows_written,
        cpci_rows_written=cpci_rows_written,
        patents_without_english_text=patents_without_english_text,
        patent_null_counts=patent_null_counts,
    )
    summary["batches_written"] = batches_written

    summary_path = args.output_dir / "build_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSummary сохранён: {summary_path}")


if __name__ == "__main__":
    main()
