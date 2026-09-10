"""Dataset entry points with lazy imports.

The strict PU metadata adapter must be importable without eagerly importing
the legacy training stack (and its optional torchvision dependencies).  The
legacy classes remain available through module-level lazy attribute access.
"""

__all__ = ["PHM2009_DG", "PU_DG"]


def __getattr__(name):
    if name == "PHM2009_DG":
        from .PHM_2009_Gearbox import PHM2009_DG

        return PHM2009_DG
    if name == "PU_DG":
        from .PU_bearing import PU_DG

        return PU_DG
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
