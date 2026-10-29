"""Errors shared by the register, gate and fetch units."""


class RegisterError(ValueError):
    """A register or licence-rules file that cannot be loaded; the message names the file and field."""


class TransitionError(ValueError):
    """A status change the register refuses."""


class LoadRefused(ValueError):
    """A record the loader will not read: wrong status, or no version to pin the text to (spec 06)."""


class SkipShareExceeded(RuntimeError):
    """Too many files in a snapshot could not be loaded; the partial result is attached as `.result`."""

    def __init__(self, message: str, result) -> None:
        super().__init__(message)
        self.result = result
