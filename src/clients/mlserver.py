import math
from urllib.parse import quote

import httpx

from src.clients.errors import UpstreamError


class MLServerClient:
    def __init__(self, http: httpx.AsyncClient, model_name: str) -> None:
        self.http = http
        self.model_name = model_name

    async def encode(self, text: str) -> list[float]:
        request = {
            "parameters": {"content_type": "pd"},
            "inputs": [
                {
                    "name": "text",
                    "shape": [1],
                    "datatype": "BYTES",
                    "parameters": {"content_type": "str"},
                    "data": [text],
                }
            ],
        }
        try:
            response = await self.http.post(
                f"/v2/models/{quote(self.model_name, safe='')}/infer", json=request
            )
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise UpstreamError("MLServer request timed out", 504) from exc
        except httpx.RequestError as exc:
            raise UpstreamError("MLServer is unavailable", 503) from exc
        except httpx.HTTPStatusError as exc:
            status = 503 if response.status_code in (404, 503) else 502
            raise UpstreamError("MLServer inference failed", status) from exc

        try:
            output = response.json()["outputs"][0]
            vector = [float(value) for value in output["data"]]
            if output["shape"] != [1, 1024] or len(vector) != 1024:
                raise ValueError("Unexpected embedding shape")
            if not all(math.isfinite(value) for value in vector):
                raise ValueError("Non-finite embedding")
            return vector
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            raise UpstreamError("MLServer returned an invalid embedding", 502) from exc
