#!/usr/bin/env python3
"""Download a resumable raw XML snapshot from EPO Open Patent Services."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import httpx
from dotenv import load_dotenv

TOKEN_URL = "https://ops.epo.org/3.2/auth/accesstoken"
SEARCH_URL = "https://ops.epo.org/3.2/rest-services/published-data/search/biblio"
PAGE_SIZE = 100
MAX_RESULTS_PER_QUERY = 2_000
DEFAULT_TARGET_GIB = 1.0
DEFAULT_DELAY_SECONDS = 5.0
MAX_RETRIES = 6


@dataclass(frozen=True)
class SearchPage:
    content: bytes
    total_results: int
    record_count: int
    quota_headers: dict[str, str]


@dataclass(frozen=True)
class AccessToken:
    value: str
    refresh_at: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download recent EPO bibliographic records as untouched XML pages."
    )
    parser.add_argument(
        "--target-gib",
        type=float,
        default=DEFAULT_TARGET_GIB,
        help="Stop after at least this many GiB have been saved (default: 1)",
    )
    parser.add_argument(
        "--start-date",
        type=date.fromisoformat,
        default=date.today(),
        help="Newest publication date in YYYY-MM-DD format (default: today)",
    )
    parser.add_argument(
        "--oldest-date",
        type=date.fromisoformat,
        default=date.today() - timedelta(days=730),
        help="Stop searching before this date (default: two years ago)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/epo"),
        help="Directory for XML pages and manifest.json",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=DEFAULT_DELAY_SECONDS,
        help="Minimum delay between OPS search requests (default: 5)",
    )
    return parser.parse_args()


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise SystemExit(f"Missing required environment variable: {name}")
    return value


def local_name(tag: str) -> str:
    return tag.rsplit("}", maxsplit=1)[-1]


def parse_search_metadata(xml: bytes) -> tuple[int, int]:
    root = ElementTree.fromstring(xml)
    search_nodes = [node for node in root.iter() if local_name(node.tag) == "biblio-search"]
    total_results = (
        int(search_nodes[0].attrib.get("total-result-count", "0")) if search_nodes else 0
    )
    record_count = sum(1 for node in root.iter() if local_name(node.tag) == "exchange-document")
    return total_results, record_count


def get_access_token(client: httpx.Client, key: str, secret: str) -> AccessToken:
    response = client.post(
        TOKEN_URL,
        auth=(key, secret),
        data={"grant_type": "client_credentials"},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    response.raise_for_status()
    token = response.json().get("access_token")
    if not token:
        raise RuntimeError("EPO token response does not contain access_token")
    expires_in = float(response.json().get("expires_in", 1_200))
    return AccessToken(value=str(token), refresh_at=time.monotonic() + max(expires_in - 60, 60))


def quota_headers(response: httpx.Response) -> dict[str, str]:
    return {
        name: value
        for name, value in response.headers.items()
        if name.lower().startswith(("x-individualquota", "x-registeredquota", "x-throttling"))
    }


def retry_delay(response: httpx.Response | None, attempt: int) -> float:
    if response is not None and (retry_after := response.headers.get("Retry-After")):
        try:
            return max(float(retry_after), 1.0)
        except ValueError:
            pass
    return min(2**attempt + random.random(), 60.0)


def fetch_page(
    client: httpx.Client,
    token: str,
    query: str,
    begin: int,
    end: int,
) -> SearchPage:
    last_error: httpx.HTTPError | None = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.get(
                SEARCH_URL,
                params={"q": query},
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/exchange+xml",
                    "X-OPS-Range": f"{begin}-{end}",
                },
            )
        except httpx.TransportError as error:
            last_error = error
            delay = retry_delay(None, attempt)
            print(f"OPS connection failed; retrying in {delay:.1f}s: {error}")
            time.sleep(delay)
            continue
        if response.status_code == httpx.codes.NOT_FOUND:
            return SearchPage(b"", 0, 0, quota_headers(response))
        if response.status_code not in {
            httpx.codes.TOO_MANY_REQUESTS,
            httpx.codes.BAD_GATEWAY,
            httpx.codes.SERVICE_UNAVAILABLE,
            httpx.codes.GATEWAY_TIMEOUT,
        }:
            response.raise_for_status()
            total_results, record_count = parse_search_metadata(response.content)
            return SearchPage(
                response.content,
                total_results,
                record_count,
                quota_headers(response),
            )

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            last_error = error
        delay = retry_delay(response, attempt)
        print(f"OPS returned HTTP {response.status_code}; retrying in {delay:.1f}s")
        time.sleep(delay)

    if last_error is None:
        raise RuntimeError("EPO request failed without an HTTP error")
    raise last_error


def empty_manifest() -> dict[str, Any]:
    return {
        "version": 1,
        "source": SEARCH_URL,
        "created_at": datetime.now(UTC).isoformat(),
        "pages": [],
    }


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return empty_manifest()
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("pages"), list):
        raise SystemExit(f"Invalid manifest: {path}")
    return value


def save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    temporary_path = path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def saved_bytes(manifest: dict[str, Any]) -> int:
    return sum(int(page.get("bytes", 0)) for page in manifest["pages"])


def page_key(publication_date: date, begin: int, end: int) -> tuple[str, int, int]:
    return publication_date.isoformat(), begin, end


def completed_pages(manifest: dict[str, Any]) -> set[tuple[str, int, int]]:
    return {
        page_key(
            date.fromisoformat(page["publication_date"]), page["range_begin"], page["range_end"]
        )
        for page in manifest["pages"]
    }


def write_page(path: Path, content: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".xml.tmp")
    temporary_path.write_bytes(content)
    temporary_path.replace(path)
    return hashlib.sha256(content).hexdigest()


def append_page(
    manifest: dict[str, Any],
    *,
    publication_date: date,
    begin: int,
    end: int,
    relative_path: Path,
    page: SearchPage,
    digest: str,
) -> None:
    manifest["pages"].append(
        {
            "publication_date": publication_date.isoformat(),
            "query": f"pd={publication_date:%Y%m%d}",
            "range_begin": begin,
            "range_end": end,
            "total_results": page.total_results,
            "record_count": page.record_count,
            "path": relative_path.as_posix(),
            "bytes": len(page.content),
            "sha256": digest,
            "downloaded_at": datetime.now(UTC).isoformat(),
            "quota_headers": page.quota_headers,
        }
    )
    manifest["updated_at"] = datetime.now(UTC).isoformat()


def validate_args(args: argparse.Namespace) -> None:
    if args.target_gib <= 0:
        raise SystemExit("--target-gib must be positive")
    if args.delay_seconds < 0:
        raise SystemExit("--delay-seconds cannot be negative")
    if args.start_date < args.oldest_date:
        raise SystemExit("--start-date must not be earlier than --oldest-date")


def download(args: argparse.Namespace) -> None:
    validate_args(args)
    key = require_env("EPO_OPS_CONSUMER_KEY")
    secret = require_env("EPO_OPS_CONSUMER_SECRET")
    target_bytes = int(args.target_gib * 1024**3)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    manifest = load_manifest(manifest_path)
    done = completed_pages(manifest)
    total_bytes = saved_bytes(manifest)

    if total_bytes >= target_bytes:
        print(f"Target already reached: {total_bytes / 1024**3:.3f} GiB")
        return

    with httpx.Client(timeout=60, follow_redirects=True) as client:
        access_token = get_access_token(client, key, secret)
        publication_date = args.start_date

        while publication_date >= args.oldest_date and total_bytes < target_bytes:
            query = f"pd={publication_date:%Y%m%d}"
            begin = 1
            total_for_date: int | None = None
            requested_page = False

            while begin <= MAX_RESULTS_PER_QUERY and total_bytes < target_bytes:
                end = min(begin + PAGE_SIZE - 1, MAX_RESULTS_PER_QUERY)
                key_value = page_key(publication_date, begin, end)
                if key_value in done:
                    matching_page = next(
                        page
                        for page in manifest["pages"]
                        if page_key(
                            date.fromisoformat(page["publication_date"]),
                            page["range_begin"],
                            page["range_end"],
                        )
                        == key_value
                    )
                    total_for_date = int(matching_page["total_results"])
                    begin += PAGE_SIZE
                    if begin > min(total_for_date, MAX_RESULTS_PER_QUERY):
                        break
                    continue

                if time.monotonic() >= access_token.refresh_at:
                    access_token = get_access_token(client, key, secret)
                page = fetch_page(client, access_token.value, query, begin, end)
                requested_page = True
                if not page.content or page.record_count == 0:
                    break

                total_for_date = page.total_results
                relative_path = (
                    Path(publication_date.isoformat()) / f"records_{begin:04d}_{end:04d}.xml"
                )
                digest = write_page(args.output_dir / relative_path, page.content)
                append_page(
                    manifest,
                    publication_date=publication_date,
                    begin=begin,
                    end=end,
                    relative_path=relative_path,
                    page=page,
                    digest=digest,
                )
                save_manifest(manifest_path, manifest)
                done.add(key_value)
                total_bytes += len(page.content)
                print(
                    f"{publication_date} {begin}-{end}: {page.record_count} records, "
                    f"{total_bytes / 1024**3:.3f}/{args.target_gib:.3f} GiB"
                )

                begin += PAGE_SIZE
                if begin > min(total_for_date, MAX_RESULTS_PER_QUERY):
                    break
                time.sleep(args.delay_seconds)

            if total_for_date is not None and total_for_date > MAX_RESULTS_PER_QUERY:
                print(
                    f"Warning: {publication_date} has {total_for_date} hits; OPS exposes only "
                    f"the first {MAX_RESULTS_PER_QUERY} for one query.",
                    file=sys.stderr,
                )
            publication_date -= timedelta(days=1)
            if requested_page and total_bytes < target_bytes:
                time.sleep(args.delay_seconds)

    if total_bytes < target_bytes:
        raise SystemExit(
            f"Stopped at --oldest-date with {total_bytes / 1024**3:.3f} GiB; "
            "choose an earlier date to continue."
        )

    print(f"Download complete: {total_bytes / 1024**3:.3f} GiB in {len(manifest['pages'])} pages")


def main() -> None:
    load_dotenv()
    args = parse_args()
    try:
        download(args)
    except httpx.HTTPStatusError as error:
        body = error.response.text[:500]
        print(f"EPO OPS returned HTTP {error.response.status_code}: {body}", file=sys.stderr)
        raise SystemExit(1) from error
    except (httpx.HTTPError, ElementTree.ParseError) as error:
        print(f"Failed to download or parse EPO OPS data: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
