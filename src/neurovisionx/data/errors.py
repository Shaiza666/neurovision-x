"""Dataset errors that do not pull in NIfTI I/O."""


class MissingBraTSDataError(FileNotFoundError):
    """Raised when official BraTS volumes are not present."""


class CorruptedNiftiError(OSError):
    """Raised when a NIfTI file exists but cannot be read as an image."""


class DatasetValidationError(ValueError):
    """Raised when a patient volume fails structural checks."""
