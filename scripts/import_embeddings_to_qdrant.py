#!/usr/bin/env python3
"""Import a complete MLflow embedding Run from RustFS into a versioned Qdrant collection.

Credentials are read from .env. No collection is deleted or recreated. Downloads
are verified by SHA256; stable UUIDs and per-part checkpoints make retries safe.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import uuid
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import boto3
import numpy as np
import polars as pl
import requests
from botocore.client import BaseClient
from botocore.config import Config
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

ROOT = Path(__file__).resolve().parents[1]
PAYLOAD_COLUMNS = [
    "publication_id",
    "family_id",
    "publication_authority",
    "document_number",
    "document_kind",
    "publication_date",
    "source_language",
    "title",
    "abstract",
    "cpc_codes",
    "ipc_codes",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def point_id(run_id: str, publication_id: str) -> str:
    return str(uuid.uuid5(uuid.UUID(run_id), publication_id))


def download_verified(client: BaseClient, uri: str, path: Path, digest: str) -> Path:
    if path.exists() and sha256_file(path) == digest:
        return path
    source = urlsplit(uri)
    if source.scheme != "s3" or not source.netloc or not source.path.strip("/"):
        raise ValueError("Expected an s3://bucket/key source")
    client.download_file(source.netloc, source.path.lstrip("/"), str(path))
    if sha256_file(path) != digest:
        raise ValueError(f"SHA256 mismatch: {uri}")
    return path


def fetch_manifest(run_id: str) -> dict[str, Any]:
    base = os.environ["MLFLOW_TRACKING_URI"].rstrip("/")
    with requests.Session() as session:
        session.auth = (
            os.environ["MLFLOW_TRACKING_USERNAME"],
            os.environ["MLFLOW_TRACKING_PASSWORD"],
        )
        response = session.get(
            f"{base}/api/2.0/mlflow/runs/get", params={"run_id": run_id}, timeout=60
        )
        response.raise_for_status()
        run = response.json()["run"]
        if run["info"]["status"] != "FINISHED":
            raise ValueError("Embedding Run must be FINISHED")
        uri = run["info"]["artifact_uri"]
        if not uri.startswith("mlflow-artifacts:/"):
            raise ValueError("Expected proxied MLflow artifacts")
        artifact_path = uri.removeprefix("mlflow-artifacts:/").lstrip("/")
        response = session.get(
            f"{base}/api/2.0/mlflow-artifacts/artifacts/{artifact_path}/embedding_manifest.json",
            timeout=60,
        )
        response.raise_for_status()
        manifest = response.json()
    expected = manifest["expected_rows"]
    if (
        manifest["run_id"] != run_id
        or not manifest["complete"]
        or expected <= 0
        or manifest["completed_rows"] != expected
        or sum(part["rows"] for part in manifest["parts"]) != expected
    ):
        raise ValueError("Embedding manifest is incomplete or inconsistent")
    part_numbers = [part["part"] for part in manifest["parts"]]
    if part_numbers != list(range(len(part_numbers))):
        raise ValueError("Duplicate or missing parts in manifest")
    return manifest


def shard_vectors(shard: pl.DataFrame, dimension: int) -> np.ndarray:
    if shard.is_empty() or shard["publication_id"].null_count():
        raise ValueError("Empty shard or null publication_id")
    if shard.filter(pl.col("embedding").is_null()).height:
        raise ValueError("Null embedding")
    if shard.filter(pl.col("embedding").list.len() != dimension).height:
        raise ValueError("Unexpected embedding dimension")
    vectors = shard["embedding"].cast(pl.Array(pl.Float32, dimension)).to_numpy()
    if not np.isfinite(vectors).all():
        raise ValueError("Non-finite embeddings")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-3):
        raise ValueError("Expected normalized embeddings")
    return vectors


def payload_rows(shard: pl.DataFrame, dataset: pl.DataFrame) -> list[dict[str, Any]]:
    joined = shard.select("publication_id").join(
        dataset, on="publication_id", how="left", validate="1:1", maintain_order="left"
    )
    if joined["title"].null_count() or joined["abstract"].null_count():
        raise ValueError("Missing patent text for an embedding")
    rows = joined.to_dicts()
    for row in rows:
        publication_date = row["publication_date"]
        if isinstance(publication_date, date):
            row["publication_date"] = f"{publication_date.isoformat()}T00:00:00Z"
    return rows


def create_client(args: argparse.Namespace) -> QdrantClient:
    key = os.environ["QDRANT_API_KEY"]
    if args.tunnel:
        # Plain gRPC only over a loopback SSH tunnel; no public insecure transport.
        return QdrantClient(
            host="127.0.0.1",
            port=18333,
            grpc_port=18334,
            https=False,
            prefer_grpc=True,
            api_key=key,
            timeout=120,
        )
    url = os.environ["QDRANT_URL"]
    parsed = urlsplit(url)
    return QdrantClient(
        url=url,
        port=parsed.port or (443 if parsed.scheme == "https" else 80),
        api_key=key,
        timeout=120,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--collection")
    parser.add_argument("--tunnel", action="store_true", help="SSH ports 18333/18334")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    manifest = fetch_manifest(args.run_id)
    if not manifest["normalize_embeddings"]:
        raise ValueError("This importer requires normalized vectors")
    dimension = manifest["embedding_dimension"]
    cache = ROOT / "data" / "qdrant_import" / args.run_id
    cache.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ["RUSTFS_ENDPOINT"],
        aws_access_key_id=os.environ["RUSTFS_ACCESS_KEY"],
        aws_secret_access_key=os.environ["RUSTFS_SECRET_KEY"],
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )
    dataset_path = download_verified(
        s3, manifest["dataset_source"], cache / "patents_search.parquet", manifest["dataset_sha256"]
    )
    dataset = pl.read_parquet(dataset_path, columns=PAYLOAD_COLUMNS)
    if dataset.height != manifest["expected_rows"]:
        raise ValueError("Dataset row count does not match manifest")
    if (
        dataset["publication_id"].null_count()
        or dataset["publication_id"].n_unique() != dataset.height
    ):
        raise ValueError("Dataset IDs must be non-null and unique")

    # Validate ALL sources before creating or writing any Qdrant collection.
    seen: set[str] = set()
    for part in manifest["parts"]:
        path = download_verified(
            s3, part["s3_uri"], cache / f"part-{part['part']:05d}.parquet", part["sha256"]
        )
        shard = pl.read_parquet(path)
        if shard.height != part["rows"]:
            raise ValueError("Shard row count mismatch")
        shard_vectors(shard, dimension)
        ids = shard["publication_id"].to_list()
        if len(set(ids)) != len(ids) or seen.intersection(ids):
            raise ValueError("Duplicate publication_id in embeddings")
        seen.update(ids)
        print(f"Verified RustFS part {part['part'] + 1}/{len(manifest['parts'])}", flush=True)
    if seen != set(dataset["publication_id"].to_list()):
        raise ValueError("Embedding IDs do not match the dataset")

    client = create_client(args)
    collection = args.collection or f"patents_bge_m3_v1_{args.run_id[:8]}"
    provenance = {
        "importer": "rustfs-parquet-v1",
        "embedding_run_id": args.run_id,
        **{
            key: manifest[key]
            for key in (
                "dataset_source",
                "dataset_sha256",
                "dataset_version",
                "model_id",
                "model_revision",
                "max_seq_length",
                "normalize_embeddings",
                "embedding_dimension",
            )
        },
    }
    if client.collection_exists(collection):
        info = client.get_collection(collection)
        checkpoint = info.config.metadata or {}
        if any(checkpoint.get(key) != value for key, value in provenance.items()):
            raise ValueError("Refusing to modify a collection with different provenance")
        config = info.config.params.vectors
        if not isinstance(config, models.VectorParams):
            raise ValueError("Expected unnamed dense vectors")
        if config.size != dimension or config.distance != models.Distance.COSINE:
            raise ValueError("Existing vector configuration does not match")
    else:
        checkpoint = {**provenance, "completed_parts": [], "import_status": "IMPORTING"}
        client.create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(
                size=dimension, distance=models.Distance.COSINE, on_disk=True
            ),
            on_disk_payload=True,
            hnsw_config=models.HnswConfigDiff(on_disk=True, max_indexing_threads=1),
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=0, max_optimization_threads=1
            ),
            metadata=checkpoint,
        )
    print(f"Collection: {collection}", flush=True)
    for field in [
        "publication_id",
        "family_id",
        "publication_authority",
        "document_kind",
        "source_language",
        "cpc_codes",
        "ipc_codes",
    ]:
        client.create_payload_index(collection, field, models.PayloadSchemaType.KEYWORD, wait=True)
    client.create_payload_index(
        collection, "publication_date", models.PayloadSchemaType.DATETIME, wait=True
    )
    completed = set(checkpoint["completed_parts"])
    for part in manifest["parts"]:
        if part["part"] in completed:
            continue
        shard = pl.read_parquet(cache / f"part-{part['part']:05d}.parquet")
        client.upload_collection(
            collection_name=collection,
            vectors=shard_vectors(shard, dimension),
            payload=payload_rows(shard, dataset),
            ids=[point_id(args.run_id, pub_id) for pub_id in shard["publication_id"].to_list()],
            batch_size=256,
            parallel=1,
            max_retries=5,
            wait=True,
        )
        completed.add(part["part"])
        checkpoint["completed_parts"] = sorted(completed)
        client.update_collection(collection, metadata=checkpoint)
        rows = sum(p["rows"] for p in manifest["parts"] if p["part"] in completed)
        print(f"Uploaded {rows:,}/{manifest['expected_rows']:,}", flush=True)
    count = client.count(collection, exact=True).count
    if count != manifest["expected_rows"]:
        raise ValueError(f"Qdrant count mismatch: {count}")
    checkpoint["import_status"] = "UPLOADED"
    checkpoint["points_count"] = count
    client.update_collection(
        collection,
        metadata=checkpoint,
        optimizers_config=models.OptimizersConfigDiff(
            indexing_threshold=20000, max_optimization_threads=1
        ),
    )
    print(f"DONE: {count:,} points; HNSW indexing enabled", flush=True)
    client.close()


if __name__ == "__main__":
    main()
