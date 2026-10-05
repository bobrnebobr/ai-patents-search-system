from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/processed/epo_v1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/eda/epo_v1"),
    )
    return parser.parse_args()


def parquet_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def save_horizontal_bar(
    frame: pd.DataFrame,
    label_column: str,
    value_column: str,
    title: str,
    output_path: Path,
) -> None:
    frame = frame.sort_values(value_column)

    figure, axis = plt.subplots(figsize=(10, 7))
    sns.barplot(
        data=frame,
        x=value_column,
        y=label_column,
        hue=label_column,
        legend=False,
        palette="viridis",
        ax=axis,
    )
    axis.set_title(title)
    axis.set_xlabel("Count")
    axis.set_ylabel("")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_publication_dates(frame: pd.DataFrame, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(11, 6))
    sns.barplot(
        data=frame,
        x="publication_date",
        y="document_count",
        color="#4C78A8",
        ax=axis,
    )
    axis.set_title("Patent publications by date")
    axis.set_xlabel("Publication date")
    axis.set_ylabel("Patent count")
    axis.tick_params(axis="x", rotation=45)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_abstract_lengths(frame: pd.DataFrame, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=frame,
        x="text_count",
        y="length_bucket",
        hue="length_bucket",
        legend=False,
        palette="magma",
        ax=axis,
    )
    axis.set_title("Abstract length distribution")
    axis.set_xlabel("Text count")
    axis.set_ylabel("Abstract length, characters")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    patents_path = parquet_path(args.input_dir / "patents.parquet")
    texts_path = parquet_path(args.input_dir / "patent_texts.parquet")
    classifications_path = parquet_path(args.input_dir / "classifications.parquet")

    connection = duckdb.connect()

    connection.execute(
        f"""
        CREATE VIEW patents AS
        SELECT *
        FROM read_parquet('{patents_path}')
        """
    )
    connection.execute(
        f"""
        CREATE VIEW patent_texts AS
        SELECT *
        FROM read_parquet('{texts_path}')
        """
    )
    connection.execute(
        f"""
        CREATE VIEW classifications AS
        SELECT *
        FROM read_parquet('{classifications_path}')
        """
    )

    authority_distribution = connection.execute(
        """
        SELECT
            COALESCE(publication_authority, 'unknown') AS publication_authority,
            COUNT(*) AS patent_count
        FROM patents
        GROUP BY publication_authority
        ORDER BY patent_count DESC
        LIMIT 15
        """
    ).df()

    language_distribution = connection.execute(
        """
        SELECT
            COALESCE(source_language, 'unknown') AS source_language,
            COUNT(*) AS text_count
        FROM patent_texts
        GROUP BY source_language
        ORDER BY text_count DESC
        """
    ).df()

    publication_dates = connection.execute(
        """
        SELECT
            CAST(publication_date AS VARCHAR) AS publication_date,
            COUNT(*) AS document_count
        FROM patents
        WHERE publication_date IS NOT NULL
        GROUP BY publication_date
        ORDER BY publication_date
        """
    ).df()

    ipc_subclasses = connection.execute(
        """
        SELECT
            SUBSTRING(classification_code, 1, 4) AS ipc_subclass,
            COUNT(*) AS classification_count
        FROM classifications
        WHERE classification_system = 'IPCR'
          AND classification_code IS NOT NULL
        GROUP BY ipc_subclass
        ORDER BY classification_count DESC
        LIMIT 20
        """
    ).df()

    abstract_lengths = connection.execute(
        """
        WITH abstract_lengths AS (
            SELECT
                LENGTH(abstract) AS length_chars
            FROM patent_texts
            WHERE abstract IS NOT NULL
              AND TRIM(abstract) != ''
        )
        SELECT
            CASE
                WHEN length_chars < 500 THEN '0–499'
                WHEN length_chars < 1_000 THEN '500–999'
                WHEN length_chars < 2_000 THEN '1,000–1,999'
                WHEN length_chars < 4_000 THEN '2,000–3,999'
                WHEN length_chars < 8_000 THEN '4,000–7,999'
                ELSE '8,000+'
            END AS length_bucket,
            CASE
                WHEN length_chars < 500 THEN 1
                WHEN length_chars < 1_000 THEN 2
                WHEN length_chars < 2_000 THEN 3
                WHEN length_chars < 4_000 THEN 4
                WHEN length_chars < 8_000 THEN 5
                ELSE 6
            END AS bucket_order,
            COUNT(*) AS text_count
        FROM abstract_lengths
        GROUP BY length_bucket, bucket_order
        ORDER BY bucket_order
        """
    ).df()

    save_horizontal_bar(
        authority_distribution,
        "publication_authority",
        "patent_count",
        "Top publication authorities",
        args.output_dir / "publication_authorities.png",
    )
    save_horizontal_bar(
        language_distribution,
        "source_language",
        "text_count",
        "Text versions by source language",
        args.output_dir / "source_languages.png",
    )
    save_publication_dates(
        publication_dates,
        args.output_dir / "publication_dates.png",
    )
    save_horizontal_bar(
        ipc_subclasses,
        "ipc_subclass",
        "classification_count",
        "Top 20 IPC subclasses",
        args.output_dir / "top_ipc_subclasses.png",
    )
    save_abstract_lengths(
        abstract_lengths,
        args.output_dir / "abstract_lengths.png",
    )

    summary = {
        "patents": int(connection.execute("SELECT COUNT(*) FROM patents").fetchone()[0]),
        "unique_families": int(
            connection.execute(
                """
                SELECT COUNT(DISTINCT family_id)
                FROM patents
                WHERE family_id IS NOT NULL
                """
            ).fetchone()[0]
        ),
        "patents_with_text": int(
            connection.execute(
                """
                SELECT COUNT(DISTINCT publication_id)
                FROM patent_texts
                """
            ).fetchone()[0]
        ),
        "patents_without_text": int(
            connection.execute(
                """
                SELECT COUNT(*)
                FROM patents AS patent
                WHERE NOT EXISTS (
                    SELECT 1
                    FROM patent_texts AS text
                    WHERE text.publication_id = patent.publication_id
                )
                """
            ).fetchone()[0]
        ),
        "text_versions": int(connection.execute("SELECT COUNT(*) FROM patent_texts").fetchone()[0]),
        "ipcr_assignments": int(
            connection.execute(
                """
                SELECT COUNT(*)
                FROM classifications
                WHERE classification_system = 'IPCR'
                """
            ).fetchone()[0]
        ),
        "unique_ipc_subclasses": int(
            connection.execute(
                """
                SELECT COUNT(DISTINCT SUBSTRING(classification_code, 1, 4))
                FROM classifications
                WHERE classification_system = 'IPCR'
                  AND classification_code IS NOT NULL
                """
            ).fetchone()[0]
        ),
    }

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"EDA artifacts saved to: {args.output_dir}")

    connection.close()


if __name__ == "__main__":
    main()
