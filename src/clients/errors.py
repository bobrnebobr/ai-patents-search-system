class UpstreamError(Exception):
    """A safe error message for a failed external service call."""

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code
