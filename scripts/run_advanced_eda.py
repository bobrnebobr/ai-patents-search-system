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
        default=Path("reports/eda/epo_v1/advanced"),
    )
    return parser.parse_args()


def parquet_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def save_count_bar(
    frame: pd.DataFrame,
    label_column: str,
    count_column: str,
    title: str,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=frame,
        x=count_column,
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


def save_language_authority_heatmap(
    frame: pd.DataFrame,
    output_path: Path,
) -> None:
    matrix = frame.pivot(
        index="publication_authority",
        columns="source_language",
        values="text_count",
    ).fillna(0)

    matrix = matrix.loc[
        matrix.sum(axis=1).sort_values(ascending=False).index
    ]

    figure, axis = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".0f",
        cmap="YlGnBu",
        linewidths=0.5,
        ax=axis,
    )
    axis.set_title("Text versions by authority and source language")
    axis.set_xlabel("Source language")
    axis.set_ylabel("Publication authority")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_long_tail(frame: pd.DataFrame, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.plot(
        frame["rank"],
        frame["classification_count"],
        color="#4C78A8",
    )
    axis.set_yscale("log")
    axis.set_title("IPC subclass frequency long tail")
    axis.set_xlabel("Subclass rank by frequency")
    axis.set_ylabel("Classification assignments, log scale")
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

    authority_language = connection.execute(
        """
        WITH top_authorities AS (
            SELECT publication_authority
            FROM patents
            GROUP BY publication_authority
            ORDER BY COUNT(*) DESC
            LIMIT 12
        )
        SELECT
            patent.publication_authority,
            text.source_language,
            COUNT(*) AS text_count
        FROM patent_texts AS text
        JOIN patents AS patent
            ON patent.publication_id = text.publication_id
        WHERE patent.publication_authority IN (
            SELECT publication_authority
            FROM top_authorities
        )
        GROUP BY patent.publication_authority, text.source_language
        """
    ).df()

    labels_per_patent = connection.execute(
        """
        WITH label_counts AS (
            SELECT
                patent.publication_id,
                COUNT(classification.classification_code) AS label_count
            FROM patents AS patent
            LEFT JOIN classifications AS classification
                ON classification.publication_id = patent.publication_id
            GROUP BY patent.publication_id
        )
        SELECT
            CASE
                WHEN label_count >= 10 THEN '10+'
                ELSE CAST(label_count AS VARCHAR)
            END AS label_bucket,
            CASE
                WHEN label_count >= 10 THEN 10
                ELSE label_count
            END AS bucket_order,
            COUNT(*) AS patent_count
        FROM label_counts
        GROUP BY label_bucket, bucket_order
        ORDER BY bucket_order
        """
    ).df()

    ipc_long_tail = connection.execute(
        """
        WITH subclass_counts AS (
            SELECT
                SUBSTRING(classification_code, 1, 4) AS ipc_subclass,
                COUNT(*) AS classification_count
            FROM classifications
            WHERE classification_system = 'IPCR'
              AND classification_code IS NOT NULL
            GROUP BY ipc_subclass
        )
        SELECT
            ROW_NUMBER() OVER (
                ORDER BY classification_count DESC
            ) AS rank,
            ipc_subclass,
            classification_count
        FROM subclass_counts
        ORDER BY rank
        """
    ).df()

    text_versions_per_patent = connection.execute(
        """
        WITH text_counts AS (
            SELECT
                patent.publication_id,
                COUNT(text.publication_id) AS text_version_count
            FROM patents AS patent
            LEFT JOIN patent_texts AS text
                ON text.publication_id = patent.publication_id
            GROUP BY patent.publication_id
        )
        SELECT
            CASE
                WHEN text_version_count >= 4 THEN '4+'
                ELSE CAST(text_version_count AS VARCHAR)
            END AS text_version_bucket,
            CASE
                WHEN text_version_count >= 4 THEN 4
                ELSE text_version_count
            END AS bucket_order,
            COUNT(*) AS patent_count
        FROM text_counts
        GROUP BY text_version_bucket, bucket_order
        ORDER BY bucket_order
        """
    ).df()

    duplicate_text_groups = connection.execute(
        """
        WITH text_hashes AS (
            SELECT
                publication_id,
                source_language,
                MD5(
                    COALESCE(title, '') || '\n' || COALESCE(abstract, '')
                ) AS text_hash
            FROM patent_texts
            WHERE COALESCE(TRIM(title), '') != ''
               OR COALESCE(TRIM(abstract), '') != ''
        )
        SELECT
            text_hash,
            source_language,
            COUNT(*) AS text_rows,
            COUNT(DISTINCT publication_id) AS distinct_patents,
            MIN(publication_id) AS example_publication_id
        FROM text_hashes
        GROUP BY text_hash, source_language
        HAVING COUNT(DISTINCT publication_id) > 1
        ORDER BY distinct_patents DESC, text_rows DESC
        """
    ).df()

    save_language_authority_heatmap(
        authority_language,
        args.output_dir / "authority_language_heatmap.png",
    )
    save_count_bar(
        labels_per_patent,
        "label_bucket",
        "patent_count",
        "IPCR labels per patent",
        args.output_dir / "ipcr_labels_per_patent.png",
    )
    save_long_tail(
        ipc_long_tail,
        args.output_dir / "ipc_subclass_long_tail.png",
    )
    save_count_bar(
        text_versions_per_patent,
        "text_version_bucket",
        "patent_count",
        "Text versions per patent",
        args.output_dir / "text_versions_per_patent.png",
    )

    duplicate_text_groups.to_csv(
        args.output_dir / "exact_duplicate_text_groups.csv",
        index=False,
    )

    summary = {
        "top_authorities_in_language_heatmap": int(
            authority_language["publication_authority"].nunique()
        ),
        "ipc_subclasses": int(len(ipc_long_tail)),
        "duplicate_text_groups": int(len(duplicate_text_groups)),
        "duplicate_text_rows": int(duplicate_text_groups["text_rows"].sum()),
        "max_text_versions_for_one_patent": int(
            connection.execute(
                """
                SELECT MAX(text_version_count)
                FROM (
                    SELECT
                        publication_id,
                        COUNT(*) AS text_version_count
                    FROM patent_texts
                    GROUP BY publication_id
                )
                """
            ).fetchone()[0]
        ),
    }

    (args.output_dir / "advanced_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Advanced EDA artifacts saved to: {args.output_dir}")

    connection.close()


if __name__ == "__main__":
    main()