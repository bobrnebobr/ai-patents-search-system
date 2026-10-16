import math
from urllib.parse import quote

import httpx

from src.clients.errors import UpstreamError
from src.schemas import PatentClassification


async def _infer(http: httpx.AsyncClient, model_name: str, payload: dict) -> httpx.Response:
    try:
        response = await http.post(f"/v2/models/{quote(model_name, safe='')}/infer", json=payload)
        response.raise_for_status()
        return response
    except httpx.TimeoutException as exc:
        raise UpstreamError("MLServer request timed out", 504) from exc
    except httpx.RequestError as exc:
        raise UpstreamError("MLServer is unavailable", 503) from exc
    except httpx.HTTPStatusError as exc:
        status = 503 if response.status_code in (404, 503) else 502
        raise UpstreamError("MLServer inference failed", status) from exc


class MLServerClient:
    def __init__(self, http: httpx.AsyncClient, model_name: str, *, batch_size: int = 8) -> None:
        self.http = http
        self.model_name = model_name
        self.batch_size = batch_size

    async def encode(self, text: str) -> list[float]:
        return (await self.encode_batch([text]))[0]

    async def encode_batch(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for start in range(0, len(texts), self.batch_size):
            vectors.extend(await self._encode_batch(texts[start : start + self.batch_size]))
        return vectors

    async def _encode_batch(self, texts: list[str]) -> list[list[float]]:
        request = {
            "parameters": {"content_type": "pd"},
            "inputs": [
                {
                    "name": "text",
                    "shape": [len(texts)],
                    "datatype": "BYTES",
                    "parameters": {"content_type": "str"},
                    "data": texts,
                }
            ],
        }
        response = await _infer(self.http, self.model_name, request)
        try:
            output = response.json()["outputs"][0]
            vector = [float(value) for value in output["data"]]
            if output["shape"] != [len(texts), 1024] or len(vector) != len(texts) * 1024:
                raise ValueError("Unexpected embedding shape")
            if not all(math.isfinite(value) for value in vector):
                raise ValueError("Non-finite embedding")
            return [vector[start : start + 1024] for start in range(0, len(vector), 1024)]
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            raise UpstreamError("MLServer returned an invalid embedding", 502) from exc


class PatentClassifierClient:
    def __init__(self, http: httpx.AsyncClient, model_name: str) -> None:
        self.http = http
        self.model_name = model_name

    async def classify(
        self, query_vector: list[float], patent_vectors: list[list[float]]
    ) -> list[PatentClassification]:
        if not patent_vectors:
            return []
        count = len(patent_vectors)
        response = await _infer(
            self.http,
            self.model_name,
            {
                "inputs": [
                    {
                        "name": "embeddings",
                        "shape": [count, 2048],
                        "datatype": "FP32",
                        "parameters": {"content_type": "np"},
                        "data": [
                            value for patent in patent_vectors for value in query_vector + patent
                        ],
                    }
                ]
            },
        )
        try:
            outputs = {item["name"]: item for item in response.json()["outputs"]}
            labels, scores = outputs["label"], outputs["p_relevant"]
            for output in [labels, scores]:
                if output["shape"] != [count, 1] or len(output["data"]) != count:
                    raise ValueError("Unexpected classification shape")
            return [
                PatentClassification(label=label, score=score)
                for label, score in zip(labels["data"], scores["data"], strict=True)
            ]
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            raise UpstreamError("MLServer returned an invalid classification", 502) from exc
