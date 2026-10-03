from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

PATENT_SCHEMA = pa.schema(
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

TEXT_SCHEMA = pa.schema(
    [
        pa.field("publication_id", pa.string()),
        pa.field("source_language", pa.string()),
        pa.field("title", pa.string()),
        pa.field("abstract", pa.string()),
    ]
)

CLASSIFICATION_SCHEMA = pa.schema(
    [
        pa.field("publication_id", pa.string()),
        pa.field("classification_system", pa.string()),
        pa.field("classification_code", pa.string()),
        pa.field("raw_classification", pa.string()),
        pa.field("sequence", pa.string()),
    ]
)


class ParquetBatchWriter:
    """Записывает Parquet порциями, не удерживая весь датасет в RAM."""

    def __init__(self, output_path: Path, schema: pa.Schema) -> None:
        self.output_path = output_path
        self.schema = schema
        self.writer: pq.ParquetWriter | None = None
        self.rows_written = 0

    def write(self, rows: list[dict[str, object | None]]) -> None:
        if not rows:
            return

        if self.writer is None:
            self.writer = pq.ParquetWriter(
                self.output_path,
                self.schema,
                compression="zstd",
            )

        table = pa.Table.from_pylist(rows, schema=self.schema)
        self.writer.write_table(table)
        self.rows_written += len(rows)

    def close(self) -> None:
        if self.writer is None:
            empty_table = pa.Table.from_pylist([], schema=self.schema)
            pq.write_table(
                empty_table,
                self.output_path,
                compression="zstd",
            )
            return

        self.writer.close()


def local_name(tag: str) -> str:
    return tag.rsplit("}", maxsplit=1)[-1]


def element_text(element: ET.Element | None) -> str | None:
    if element is None:
        return None

    parts = [part.strip() for part in element.itertext() if part and part.strip()]
    return " ".join(parts) or None


def find_first(element: ET.Element | None, tag_name: str) -> ET.Element | None:
    if element is None:
        return None

    return next(
        (node for node in element.iter() if local_name(node.tag) == tag_name),
        None,
    )


def find_all(element: ET.Element, tag_name: str) -> list[ET.Element]:
    return [node for node in element.iter() if local_name(node.tag) == tag_name]


def child_text(element: ET.Element | None, tag_name: str) -> str | None:
    if element is None:
        return None

    child = next(
        (node for node in element if local_name(node.tag) == tag_name),
        None,
    )
    return element_text(child)


def document_id_by_type(
    element: ET.Element | None,
    document_id_type: str,
) -> ET.Element | None:
    if element is None:
        return None

    return next(
        (
            node
            for node in find_all(element, "document-id")
            if node.attrib.get("document-id-type") == document_id_type
        ),
        None,
    )


def parse_date(raw_date: str | None) -> date | None:
    if raw_date is None:
        return None

    try:
        return datetime.strptime(raw_date, "%Y%m%d").date()
    except ValueError:
        return None


def source_language_of(element: ET.Element) -> str:
    return (
        element.attrib.get("lang")
        or element.attrib.get("{http://www.w3.org/XML/1998/namespace}lang")
        or "und"
    )


def merge_text(existing: str | None, new_value: str) -> str:
    if existing is None:
        return new_value
    if new_value in existing:
        return existing
    return f"{existing}\n{new_value}"


def normalize_ipc(raw_value: str) -> str:
    compact = re.sub(r"\s+", "", raw_value)
    match = re.match(r"[A-HY]\d{2}[A-Z]\d{1,4}/\d{1,6}", compact)
    return match.group(0) if match else compact


def parse_document(
    document: ET.Element,
) -> tuple[
    dict[str, object | None],
    list[dict[str, object | None]],
    list[dict[str, object | None]],
]:
    bibliographic_data = find_first(document, "bibliographic-data")
    publication_reference = find_first(bibliographic_data, "publication-reference")

    docdb_id = document_id_by_type(publication_reference, "docdb")
    epodoc_id = document_id_by_type(publication_reference, "epodoc")

    publication_authority = document.attrib.get("country") or child_text(docdb_id, "country")
    document_number = document.attrib.get("doc-number") or child_text(docdb_id, "doc-number")
    document_kind = document.attrib.get("kind") or child_text(docdb_id, "kind")
    publication_date_raw = child_text(docdb_id, "date")

    publication_id = child_text(epodoc_id, "doc-number")
    if publication_id is None:
        publication_id = "".join(
            value
            for value in [
                publication_authority,
                document_number,
                document_kind,
            ]
            if value
        )

    patent = {
        "publication_id": publication_id,
        "family_id": document.attrib.get("family-id"),
        "publication_authority": publication_authority,
        "document_number": document_number,
        "document_kind": document_kind,
        "publication_date": parse_date(publication_date_raw),
        "publication_date_raw": publication_date_raw,
    }

    texts_by_language: dict[str, dict[str, str | None]] = defaultdict(
        lambda: {"title": None, "abstract": None},
    )

    for xml_tag, column_name in {
        "invention-title": "title",
        "abstract": "abstract",
    }.items():
        for node in find_all(document, xml_tag):
            value = element_text(node)
            if value is None:
                continue

            source_language = source_language_of(node)
            texts_by_language[source_language][column_name] = merge_text(
                texts_by_language[source_language][column_name],
                value,
            )

    text_rows = [
        {
            "publication_id": publication_id,
            "source_language": source_language,
            "title": values["title"],
            "abstract": values["abstract"],
        }
        for source_language, values in texts_by_language.items()
    ]

    classification_rows = []
    for node in find_all(document, "classification-ipcr"):
        raw_value = element_text(node)
        if raw_value is None:
            continue

        classification_rows.append(
            {
                "publication_id": publication_id,
                "classification_system": "IPCR",
                "classification_code": normalize_ipc(raw_value),
                "raw_classification": raw_value,
                "sequence": node.attrib.get("sequence"),
            }
        )

    return patent, text_rows, classification_rows


def flush_batch(
    patent_rows: list[dict[str, object | None]],
    text_rows: list[dict[str, object | None]],
    classification_rows: list[dict[str, object | None]],
    patents_writer: ParquetBatchWriter,
    texts_writer: ParquetBatchWriter,
    classifications_writer: ParquetBatchWriter,
) -> bool:
    has_rows = bool(patent_rows or text_rows or classification_rows)

    patents_writer.write(patent_rows)
    texts_writer.write(text_rows)
    classifications_writer.write(classification_rows)

    patent_rows.clear()
    text_rows.clear()
    classification_rows.clear()

    return has_rows


def build_dataset(
    input_path: Path,
    output_dir: Path,
    batch_size: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    patents_writer = ParquetBatchWriter(
        output_dir / "patents.parquet",
        PATENT_SCHEMA,
    )
    texts_writer = ParquetBatchWriter(
        output_dir / "patent_texts.parquet",
        TEXT_SCHEMA,
    )
    classifications_writer = ParquetBatchWriter(
        output_dir / "classifications.parquet",
        CLASSIFICATION_SCHEMA,
    )

    patent_rows: list[dict[str, object | None]] = []
    text_rows: list[dict[str, object | None]] = []
    classification_rows: list[dict[str, object | None]] = []

    seen_publication_ids: set[str] = set()
    null_counts = {field.name: 0 for field in PATENT_SCHEMA}

    documents_seen = 0
    documents_without_id = 0
    duplicates_skipped = 0
    patents_without_any_text = 0
    batches_written = 0

    context = ET.iterparse(input_path, events=("start", "end"))
    _, root = next(context)

    for event, document in context:
        if event != "end" or local_name(document.tag) != "exchange-document":
            continue

        documents_seen += 1
        patent, current_text_rows, current_classification_rows = parse_document(
            document,
        )
        publication_id = patent["publication_id"]

        if not publication_id:
            documents_without_id += 1
        elif publication_id in seen_publication_ids:
            duplicates_skipped += 1
        else:
            seen_publication_ids.add(str(publication_id))
            patent_rows.append(patent)
            text_rows.extend(current_text_rows)
            classification_rows.extend(current_classification_rows)

            if not current_text_rows:
                patents_without_any_text += 1

            for column, value in patent.items():
                if value is None:
                    null_counts[column] += 1

        # Освобождаем XML-элемент и ссылки на уже обработанные документы.
        document.clear()
        root.clear()

        if documents_seen % batch_size == 0:
            if flush_batch(
                patent_rows,
                text_rows,
                classification_rows,
                patents_writer,
                texts_writer,
                classifications_writer,
            ):
                batches_written += 1

            print(f"Обработано документов: {documents_seen:,}")

    if flush_batch(
        patent_rows,
        text_rows,
        classification_rows,
        patents_writer,
        texts_writer,
        classifications_writer,
    ):
        batches_written += 1

    patents_writer.close()
    texts_writer.close()
    classifications_writer.close()

    summary = {
        "source_file": str(input_path),
        "batch_size": batch_size,
        "batches_written": batches_written,
        "documents_seen": documents_seen,
        "documents_without_id": documents_without_id,
        "duplicates_skipped": duplicates_skipped,
        "patents_written": patents_writer.rows_written,
        "patent_text_rows_written": texts_writer.rows_written,
        "classification_rows_written": classifications_writer.rows_written,
        "patents_without_any_text": patents_without_any_text,
        "patent_null_counts": null_counts,
    }

    (output_dir / "cleaning_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/epo_patents.xml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/v1"),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5_000,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_dataset(
        input_path=args.input,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )
