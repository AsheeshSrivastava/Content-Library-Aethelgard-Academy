"""Errors shared by the register, gate and fetch units."""


class RegisterError(ValueError):
    """A register or licence-rules file that cannot be loaded; the message names the file and field."""


class TransitionError(ValueError):
    """A status change the register refuses."""
