#!/usr/bin/env python3
"""Upload a local dataset to an S3-compatible RustFS bucket and verify its size."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from dotenv import load_dotenv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload a dataset file to RustFS.")
    parser.add_argument(
        "--file",
        type=Path,
        default=Path("data/exports/epo_patents_1g.xml"),
        help="Local file to upload",
    )
    parser.add_argument(
        "--object-key",
        default="epo/epo_patents_1g.xml",
        help="Object key inside the target bucket",
    )
    return parser.parse_args()


def require_env(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if not value:
        raise SystemExit(f"Missing required environment variable: {name}")
    return value


def run_mc(arguments: list[str], environment: dict[str, str], *, capture: bool = False) -> str:
    result = subprocess.run(
        ["mc", *arguments],
        env=environment,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
    )
    return result.stdout if capture else ""


def remote_size(target: str, environment: dict[str, str]) -> int:
    output = run_mc(["stat", "--json", target], environment, capture=True)
    responses = [json.loads(line) for line in output.splitlines() if line.strip()]
    if not responses:
        raise RuntimeError(f"RustFS returned no metadata for {target}")
    size = responses[-1].get("size")
    if not isinstance(size, int):
        raise RuntimeError(f"RustFS metadata does not contain an integer size for {target}")
    return size


def upload(file_path: Path, object_key: str) -> str:
    if shutil.which("mc") is None:
        raise SystemExit("The MinIO-compatible 'mc' client is not installed")
    if not file_path.is_file():
        raise SystemExit(f"Dataset file does not exist: {file_path}")
    if file_path.stat().st_size == 0:
        raise SystemExit(f"Dataset file is empty: {file_path}")

    endpoint = require_env("RUSTFS_ENDPOINT")
    access_key = require_env("RUSTFS_ACCESS_KEY")
    secret_key = require_env("RUSTFS_SECRET_KEY")
    bucket = require_env("RUSTFS_BUCKET", "datasets")
    normalized_key = object_key.strip("/")
    if not normalized_key:
        raise SystemExit("--object-key must not be empty")

    alias = "rustfs-upload"
    target = f"{alias}/{bucket}/{normalized_key}"
    with tempfile.TemporaryDirectory(prefix="rustfs-mc-") as config_dir:
        environment = os.environ.copy()
        environment["MC_CONFIG_DIR"] = config_dir
        run_mc(
            [
                "alias",
                "set",
                alias,
                endpoint,
                access_key,
                secret_key,
                "--api",
                "S3v4",
                "--path",
                "on",
            ],
            environment,
            capture=True,
        )
        run_mc(["mb", "--ignore-existing", f"{alias}/{bucket}"], environment, capture=True)
        run_mc(
            [
                "cp",
                "--attr",
                "Content-Type=application/xml",
                "--max-workers",
                "4",
                str(file_path),
                target,
            ],
            environment,
        )
        uploaded_size = remote_size(target, environment)

    local_size = file_path.stat().st_size
    if uploaded_size != local_size:
        raise RuntimeError(
            f"Upload size mismatch: local={local_size} bytes, RustFS={uploaded_size} bytes"
        )
    return f"s3://{bucket}/{normalized_key}"


def main() -> None:
    load_dotenv()
    load_dotenv(".env.rustfs")
    args = parse_args()
    destination = upload(args.file, args.object_key)
    print(f"Upload verified: {destination} ({args.file.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
