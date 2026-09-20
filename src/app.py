from litestar import Litestar

from src.api.health import healthz
from src.api.version import version

app = Litestar(route_handlers=[healthz, version])