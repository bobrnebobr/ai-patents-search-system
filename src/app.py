from litestar import Litestar

from src.api.health import health, healthz
from src.api.version import version

app = Litestar(route_handlers=[healthz, health, version])