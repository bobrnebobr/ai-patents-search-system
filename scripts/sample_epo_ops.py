#!/usr/bin/env python3
"""Download a small bibliographic sample from EPO Open Patent Services."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import httpx

TOKEN_URL = "https://ops.epo.org/3.2/auth/accesstoken"
SEARCH_URL = "https://ops.epo.org/3.2/rest-services/published-data/search/biblio"
DEFAULT_QUERY = 'cpc=G06N and pd within "20240101 20261231"'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a small EPO OPS XML sample and convert it to JSONL."
    )
    parser.add_argument("--query", default=DEFAULT_QUERY, help="OPS CQL search query")
    parser.add_argument("--limit", type=int, default=10, help="Number of patents (1-100)")
    parser.add_argument("--output-dir", type=Path, default=Path("data/samples/epo_ops"))
    return parser.parse_args()


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise SystemExit(f"Missing required environment variable: {name}")
    return value


def get_access_token(client: httpx.Client, key: str, secret: str) -> str:
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
    return str(token)


def download_sample(client: httpx.Client, token: str, query: str, limit: int) -> bytes:
    response = client.get(
        SEARCH_URL,
        params={"q": query, "Range": f"1-{limit}"},
        headers={"Authorization": f"Bearer {token}", "Accept": "application/xml"},
    )
    response.raise_for_status()
    return response.content


def local_name(tag: str) -> str:
    return tag.rsplit("}", maxsplit=1)[-1]


def text_content(element: ElementTree.Element) -> str:
    return " ".join(" ".join(element.itertext()).split())


def descendants(element: ElementTree.Element, name: str) -> list[ElementTree.Element]:
    return [child for child in element.iter() if local_name(child.tag) == name]


def localized_values(element: ElementTree.Element, name: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for child in descendants(element, name):
        text = text_content(child)
        if text:
            language = child.attrib.get("lang") or child.attrib.get("language") or "unknown"
            values.setdefault(language, text)
    return values


def party_names(element: ElementTree.Element, party: str) -> list[str]:
    result: list[str] = []
    for node in descendants(element, party):
        names = descendants(node, "name")
        if names and (value := text_content(names[0])) and value not in result:
            result.append(value)
    return result


def classification_codes(element: ElementTree.Element, kind: str) -> list[str]:
    result: list[str] = []
    for node in descendants(element, kind):
        text_nodes = descendants(node, "text")
        value = text_content(text_nodes[0] if text_nodes else node).replace(" ", "")
        if value and value not in result:
            result.append(value)
    return result


def cpc_codes(element: ElementTree.Element) -> list[str]:
    result = classification_codes(element, "classification-cpc")
    for node in descendants(element, "patent-classification"):
        schemes = descendants(node, "classification-scheme")
        if not schemes or not schemes[0].attrib.get("scheme", "").startswith("CPC"):
            continue
        parts: dict[str, str] = {}
        for child in node:
            name = local_name(child.tag)
            if name in {"section", "class", "subclass", "main-group", "subgroup"}:
                parts[name] = text_content(child)
        required = {"section", "class", "subclass", "main-group", "subgroup"}
        if not required.issubset(parts):
            continue
        value = (
            f"{parts['section']}{parts['class']}{parts['subclass']}"
            f"{parts['main-group']}/{parts['subgroup']}"
        )
        if value not in result:
            result.append(value)
    return result


def reference_date(element: ElementTree.Element, reference_name: str) -> str | None:
    references = descendants(element, reference_name)
    if not references:
        return None
    dates = descendants(references[0], "date")
    return text_content(dates[0]) if dates else None


def parse_exchange_document(document: ElementTree.Element) -> dict[str, Any]:
    country = document.attrib.get("country")
    number = document.attrib.get("doc-number")
    kind = document.attrib.get("kind")
    publication_number = "".join(part for part in (country, number, kind) if part)

    return {
        "publication_number": publication_number or None,
        "country": country,
        "document_number": number,
        "kind": kind,
        "family_id": document.attrib.get("family-id"),
        "publication_date": reference_date(document, "publication-reference"),
        "application_date": reference_date(document, "application-reference"),
        "titles": localized_values(document, "invention-title"),
        "abstracts": localized_values(document, "abstract"),
        "cpc_codes": cpc_codes(document),
        "ipc_codes": classification_codes(document, "classification-ipcr"),
        "applicants": party_names(document, "applicant"),
        "inventors": party_names(document, "inventor"),
    }


def parse_records(xml: bytes) -> list[dict[str, Any]]:
    root = ElementTree.fromstring(xml)
    documents = descendants(root, "exchange-document")
    return [parse_exchange_document(document) for document in documents]


def write_outputs(output_dir: Path, xml: bytes, records: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    xml_path = output_dir / "sample.xml"
    jsonl_path = output_dir / "sample.jsonl"
    xml_path.write_bytes(xml)
    with jsonl_path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved raw response: {xml_path}")
    print(f"Saved {len(records)} parsed records: {jsonl_path}")


def main() -> None:
    args = parse_args()
    if not 1 <= args.limit <= 100:
        raise SystemExit("--limit must be between 1 and 100")

    key = require_env("EPO_OPS_CONSUMER_KEY")
    secret = require_env("EPO_OPS_CONSUMER_SECRET")

    try:
        with httpx.Client(timeout=30, follow_redirects=True) as client:
            token = get_access_token(client, key, secret)
            xml = download_sample(client, token, args.query, args.limit)
        records = parse_records(xml)
        write_outputs(args.output_dir, xml, records)
    except httpx.HTTPStatusError as error:
        body = error.response.text[:500]
        print(f"EPO OPS returned HTTP {error.response.status_code}: {body}", file=sys.stderr)
        raise SystemExit(1) from error
    except (httpx.HTTPError, ElementTree.ParseError) as error:
        print(f"Failed to download or parse EPO OPS data: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
