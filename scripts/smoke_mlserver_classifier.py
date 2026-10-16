"""Check encoder -> classifier inference over the local MLServer V2 API."""

import argparse

import httpx
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    args = parser.parse_args()
    with httpx.Client(base_url=args.url, timeout=120, trust_env=False) as client:
        for name in ["patent-bi-encoder", "patent-classifier"]:
            client.get(f"/v2/models/{name}/ready").raise_for_status()
        texts = [
            "Methods for recycling spent lithium-ion batteries",
            "Battery recycling. Recovery of lithium and nickel from spent batteries.",
        ]
        encoded = client.post(
            "/v2/models/patent-bi-encoder/infer",
            json={
                "parameters": {"content_type": "pd"},
                "inputs": [
                    {
                        "name": "text",
                        "shape": [2],
                        "datatype": "BYTES",
                        "data": texts,
                        "parameters": {"content_type": "str"},
                    }
                ],
            },
        )
        encoded.raise_for_status()
        output = encoded.json()["outputs"][0]
        vectors = np.asarray(output["data"], dtype=np.float32).reshape(output["shape"])
        if vectors.shape != (2, 1024) or not np.isfinite(vectors).all():
            raise ValueError("Unexpected encoder output")
        response = client.post(
            "/v2/models/patent-classifier/infer",
            json={
                "inputs": [
                    {
                        "name": "embeddings",
                        "shape": [1, 2048],
                        "datatype": "FP32",
                        "data": vectors.reshape(-1).tolist(),
                        "parameters": {"content_type": "np"},
                    }
                ],
            },
        )
        response.raise_for_status()
        print("Classifier inference:", response.json())


if __name__ == "__main__":
    main()
