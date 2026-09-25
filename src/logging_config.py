import sys

from loguru import logger

_FORMAT = (
    "{time:YYYY-MM-DDTHH:mm:ss.SSS} | {level: <8} | {extra[request_id]: <8} | {name} | {message}"
)


def setup_logging(level: str) -> None:
    logger.remove()
    logger.configure(extra={"request_id": "-"})
    logger.add(
        sys.stdout,
        level=level.upper(),
        format=_FORMAT,
    )
