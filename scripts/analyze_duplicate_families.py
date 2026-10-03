from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed/epo_v1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/eda/epo_v1/advanced"),
    )
    return parser.parse_args()


def sql_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace("'", "''")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    patents_path = sql_path(args.data_dir / "patents.parquet")
    texts_path = sql_path(args.data_dir / "patent_texts.parquet")

    connection = duckdb.connect()
    connection.execute(
        f"""
        CREATE VIEW patents AS
        SELECT * FROM read_parquet('{patents_path}');

        CREATE VIEW patent_texts AS
        SELECT * FROM read_parquet('{texts_path}');
        """
    )

    duplicate_groups = connection.execute(
        """
        WITH text_rows AS (
            SELECT
                publication_id,
                md5(
                    coalesce(title, '') || chr(10) || coalesce(abstract, '')
                ) AS text_hash
            FROM patent_texts
            WHERE
                coalesce(trim(title), '') != ''
                OR coalesce(trim(abstract), '') != ''
        ),
        group_members AS (
            SELECT
                text_rows.text_hash,
                text_rows.publication_id,
                patents.family_id
            FROM text_rows
            INNER JOIN patents USING (publication_id)
        ),
        duplicate_groups AS (
            SELECT
                text_hash,
                count(*) AS text_rows,
                count(DISTINCT publication_id) AS distinct_publications,
                count(DISTINCT family_id) AS distinct_families,
                min(publication_id) AS example_publication_id,
                min(family_id) AS example_family_id
            FROM group_members
            GROUP BY text_hash
            HAVING count(DISTINCT publication_id) > 1
        )
        SELECT
            *,
            CASE
                WHEN distinct_families = 1 THEN 'same_family'
                WHEN distinct_families > 1 THEN 'cross_family'
                ELSE 'missing_family_id'
            END AS duplicate_scope
        FROM duplicate_groups
        ORDER BY distinct_families DESC, distinct_publications DESC;
        """
    ).fetchdf()

    duplicate_groups.to_csv(
        args.output_dir / "duplicate_family_groups.csv",
        index=False,
    )

    scopes = ["same_family", "cross_family", "missing_family_id"]
    by_scope: dict[str, dict[str, int]] = {}

    for scope in scopes:
        group = duplicate_groups[duplicate_groups["duplicate_scope"] == scope]
        by_scope[scope] = {
            "groups": int(len(group)),
            "text_rows": int(group["text_rows"].sum()) if not group.empty else 0,
            "publications": (
                int(group["distinct_publications"].sum()) if not group.empty else 0
            ),
        }

    total_groups = int(len(duplicate_groups))
    cross_family_groups = by_scope["cross_family"]["groups"]

    summary = {
        "duplicate_groups": total_groups,
        "duplicate_text_rows": int(duplicate_groups["text_rows"].sum()),
        "duplicate_publications": int(
            duplicate_groups["distinct_publications"].sum()
        ),
        "cross_family_group_share": (
            round(cross_family_groups / total_groups, 4) if total_groups else 0.0
        ),
        "by_scope": by_scope,
    }

    with (args.output_dir / "duplicate_family_summary.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nДетали групп: {args.output_dir / 'duplicate_family_groups.csv'}")


if __name__ == "__main__":
    main()