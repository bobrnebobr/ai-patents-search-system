#!/usr/bin/env python3
"""Download English WO/US patent records from EPO OPS into filtered XML pages."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import httpx
from dotenv import load_dotenv
from download_epo_ops_xml import (
    MAX_RESULTS_PER_QUERY,
    PAGE_SIZE,
    SearchPage,
    fetch_page,
    get_access_token,
    local_name,
    require_env,
)

EXCHANGE_NAMESPACE = "http://www.epo.org/exchange"
OPS_NAMESPACE = "http://ops.epo.org"
XLINK_NAMESPACE = "http://www.w3.org/1999/xlink"
DEFAULT_AUTHORITIES = ("WO", "US")

ElementTree.register_namespace("", EXCHANGE_NAMESPACE)
ElementTree.register_namespace("ops", OPS_NAMESPACE)
ElementTree.register_namespace("xlink", XLINK_NAMESPACE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download English WO/US patent records as XML pages."
    )
    parser.add_argument("--target-gib", type=float, default=1.0)
    parser.add_argument("--start-date", type=date.fromisoformat, default=date.today())
    parser.add_argument(
        "--oldest-date",
        type=date.fromisoformat,
        default=date.today() - timedelta(days=730),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/epo_en"))
    parser.add_argument("--delay-seconds", type=float, default=10.0)
    parser.add_argument("--authorities", nargs="+", default=list(DEFAULT_AUTHORITIES))
    return parser.parse_args()


def contains_english_text(document: ElementTree.Element, element_name: str) -> bool:
    return any(
        local_name(node.tag) == element_name
        and node.attrib.get("lang") == "en"
        and "".join(node.itertext()).strip()
        for node in document.iter()
    )


def filter_english_page(page: SearchPage) -> tuple[bytes, int]:
    root = ElementTree.fromstring(page.content)
    kept = 0
    for container in root.iter():
        if local_name(container.tag) != "exchange-documents":
            continue
        for document in list(container):
            if local_name(document.tag) != "exchange-document":
                continue
            if contains_english_text(document, "invention-title") and contains_english_text(
                document, "abstract"
            ):
                kept += 1
            else:
                container.remove(document)
    for node in root.iter():
        if local_name(node.tag) == "biblio-search":
            node.attrib["publications-count"] = str(kept)
    return ElementTree.tostring(root, encoding="UTF-8", xml_declaration=True), kept


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "version": 1,
            "source": "EPO OPS published-data search/biblio",
            "filters": {
                "authorities": list(DEFAULT_AUTHORITIES),
                "required_title_language": "en",
                "required_abstract_language": "en",
            },
            "created_at": datetime.now(UTC).isoformat(),
            "pages": [],
        }
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("pages"), list):
        raise SystemExit(f"Invalid manifest: {path}")
    return manifest


def save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_page(path: Path, content: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".xml.tmp")
    temporary.write_bytes(content)
    temporary.replace(path)
    return hashlib.sha256(content).hexdigest()


def page_key(query: str, begin: int, end: int) -> tuple[str, int, int]:
    return query, begin, end


def download(args: argparse.Namespace) -> None:
    if args.target_gib <= 0 or args.delay_seconds < 0:
        raise SystemExit("Target must be positive and delay must be non-negative")
    if args.start_date < args.oldest_date:
        raise SystemExit("--start-date must not be earlier than --oldest-date")

    key = require_env("EPO_OPS_CONSUMER_KEY")
    secret = require_env("EPO_OPS_CONSUMER_SECRET")
    target_bytes = int(args.target_gib * 1024**3)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    manifest = load_manifest(manifest_path)
    done = {
        page_key(page["query"], page["range_begin"], page["range_end"])
        for page in manifest["pages"]
    }
    total_bytes = sum(int(page.get("bytes", 0)) for page in manifest["pages"])
    if total_bytes >= target_bytes:
        print(f"Target already reached: {total_bytes / 1024**3:.3f} GiB")
        return

    with httpx.Client(timeout=60, follow_redirects=True) as client:
        access_token = get_access_token(client, key, secret)
        publication_date = args.start_date
        while publication_date >= args.oldest_date and total_bytes < target_bytes:
            for authority_value in args.authorities:
                authority = authority_value.upper()
                query = f"pd={publication_date:%Y%m%d} and pn={authority}"
                begin = 1
                total_for_query: int | None = None
                while begin <= MAX_RESULTS_PER_QUERY and total_bytes < target_bytes:
                    end = min(begin + PAGE_SIZE - 1, MAX_RESULTS_PER_QUERY)
                    key_value = page_key(query, begin, end)
                    if key_value in done:
                        previous = next(
                            item
                            for item in manifest["pages"]
                            if page_key(item["query"], item["range_begin"], item["range_end"])
                            == key_value
                        )
                        total_for_query = int(previous["total_results"])
                        begin += PAGE_SIZE
                        if begin > min(total_for_query, MAX_RESULTS_PER_QUERY):
                            break
                        continue

                    if time.monotonic() >= access_token.refresh_at:
                        access_token = get_access_token(client, key, secret)
                    source_page = fetch_page(client, access_token.value, query, begin, end)
                    if not source_page.content or source_page.record_count == 0:
                        break
                    total_for_query = source_page.total_results
                    content, kept_records = filter_english_page(source_page)
                    relative_path = (
                        Path(publication_date.isoformat())
                        / authority
                        / f"records_{begin:04d}_{end:04d}.xml"
                    )
                    digest = write_page(args.output_dir / relative_path, content)
                    manifest["pages"].append(
                        {
                            "publication_date": publication_date.isoformat(),
                            "authority": authority,
                            "query": query,
                            "range_begin": begin,
                            "range_end": end,
                            "total_results": source_page.total_results,
                            "source_record_count": source_page.record_count,
                            "record_count": kept_records,
                            "path": relative_path.as_posix(),
                            "bytes": len(content),
                            "sha256": digest,
                            "downloaded_at": datetime.now(UTC).isoformat(),
                            "quota_headers": source_page.quota_headers,
                        }
                    )
                    manifest["updated_at"] = datetime.now(UTC).isoformat()
                    save_manifest(manifest_path, manifest)
                    done.add(key_value)
                    total_bytes += len(content)
                    print(
                        f"{publication_date} {authority} {begin}-{end}: "
                        f"{kept_records}/{source_page.record_count} English records, "
                        f"{total_bytes / 1024**3:.3f}/{args.target_gib:.3f} GiB"
                    )
                    begin += PAGE_SIZE
                    if begin > min(total_for_query, MAX_RESULTS_PER_QUERY):
                        break
                    time.sleep(args.delay_seconds)

                if total_for_query is not None and total_for_query > MAX_RESULTS_PER_QUERY:
                    print(
                        f"Warning: query {query!r} has {total_for_query} hits; "
                        f"only the first {MAX_RESULTS_PER_QUERY} are accessible."
                    )
                if total_bytes < target_bytes:
                    time.sleep(args.delay_seconds)
            publication_date -= timedelta(days=1)

    if total_bytes < target_bytes:
        raise SystemExit(f"Stopped at --oldest-date with {total_bytes / 1024**3:.3f} GiB")
    print(f"Download complete: {total_bytes / 1024**3:.3f} GiB in {len(manifest['pages'])} pages")


def main() -> None:
    load_dotenv()
    args = parse_args()
    try:
        download(args)
    except httpx.HTTPStatusError as error:
        body = error.response.text[:500]
        print(f"EPO OPS returned HTTP {error.response.status_code}: {body}")
        raise SystemExit(1) from error
    except (httpx.HTTPError, ElementTree.ParseError) as error:
        print(f"Failed to download or parse EPO OPS data: {error}")
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
