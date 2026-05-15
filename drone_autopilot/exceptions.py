"""Project-specific exceptions."""


class MissingOptionalDependencyError(RuntimeError):
    """Raised when an optional dependency is required for the requested action."""

    def __init__(self, package: str, extra: str | None = None) -> None:
        hint = f" Install with: python3 -m pip install -e '.[{extra}]'" if extra else ""
        super().__init__(f"Missing optional dependency '{package}'.{hint}")
