import os
from pathlib import Path

os.environ["MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD"] = "false"
os.environ["MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD"] = "false"
os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "1800"

import mlflow
import numpy as np
import torch
from mlflow import MlflowClient
from mlflow.models import infer_signature

TRACKING_URI = "URI"
MODEL_NAME = "patent-classifier"
ALIAS = "champion"
WORK_DIR = Path("...")
# os.environ["MLFLOW_TRACKING_USERNAME"] = "..."
# os.environ["MLFLOW_TRACKING_PASSWORD"] = "..."

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("patent-classifier")

ckpt = torch.load(WORK_DIR / "head.pt", map_location="cpu")
sd, arch = ckpt["state_dict"], ckpt["arch"]
print("архитектура:", arch)
head_path = WORK_DIR / "head.pt"
meta_path = WORK_DIR / "head_meta.json"

# 2. Пример входа, эталонный ответ torch и сигнатура
rng = np.random.default_rng(0)
dim = arch["dim"]


def unit(n):
    a = rng.normal(size=(n, dim)).astype("float32")
    return a / np.linalg.norm(a, axis=1, keepdims=True)


example = {"embeddings": np.concatenate([unit(4), unit(4)], axis=1)}  # [4, 2048] = [u | v]

net = torch.nn.Sequential(
    torch.nn.Linear(2 * dim + 1, arch["hidden"]),
    torch.nn.ReLU(),
    torch.nn.Dropout(arch["dropout"]),
    torch.nn.Linear(arch["hidden"], arch["n_classes"]),
)
net.load_state_dict({k[len("net.") :]: v for k, v in sd.items()})
net.eval()
with torch.no_grad():
    u, v = (
        torch.tensor(example["embeddings"][:, :dim]),
        torch.tensor(example["embeddings"][:, dim:]),
    )
    cos = torch.nn.functional.cosine_similarity(u, v, dim=-1).unsqueeze(-1)
    ref = torch.softmax(net(torch.cat([(u - v).abs(), u * v, cos], -1)), -1).numpy()

import pandas as pd

example_output = pd.DataFrame(
    {
        "p_not_relevant": ref[:, 0],
        "p_uncertain": ref[:, 1],
        "p_relevant": ref[:, 2],
        "label": ["not_relevant"] * len(ref),
    }
)
signature = infer_signature(example, example_output)
print(signature)

# 3. Логирование и регистрация
with mlflow.start_run(run_name="patent-classifier-head"):
    import json

    meta = json.load(open(meta_path))
    mlflow.log_param("threshold_relevant", meta["threshold_relevant"])
    mlflow.log_param("features", meta.get("features"))
    mlflow.log_param("patent_text_format", meta.get("patent_text_format"))
    info = mlflow.pyfunc.log_model(
        name="model",
        python_model=str(Path(__file__).with_name("head_model.py")),
        artifacts={"head": str(head_path), "meta": str(meta_path)},
        signature=signature,
        input_example=example,
        pip_requirements=["mlflow==3.16.1", "torch==2.14.1", "numpy==2.5.3", "pandas==3.0.6"],
        registered_model_name=MODEL_NAME,
    )

client = MlflowClient()
version = max(int(v.version) for v in client.search_model_versions(f"name='{MODEL_NAME}'"))
client.set_registered_model_alias(MODEL_NAME, ALIAS, version)
print(f"Зарегистрировано: {MODEL_NAME} v{version}, алиас @{ALIAS}")
