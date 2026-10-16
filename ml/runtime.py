"""Compatibility for Pandas 3 string outputs in MLServer's MLflow runtime."""

import pandas as pd
from mlserver.codecs import RequestCodecLike
from mlserver.types import InferenceResponse
from mlserver_mlflow import MLflowRuntime


class PandasMLflowRuntime(MLflowRuntime):
    def encode_response(
        self, payload: object, default_codec: RequestCodecLike | None = None
    ) -> InferenceResponse:
        if isinstance(payload, pd.DataFrame):
            # MLServer 1.7 understands object strings, but not Pandas 3's StringDtype.
            columns = payload.select_dtypes(include="string").columns
            payload = payload.astype(dict.fromkeys(columns, object))
        return super().encode_response(payload, default_codec=default_codec)
