"""mergeplg: merging of point, line, and gridded rainfall data.

The package provides interpolation and merging routines to combine
point (gauge) and line (CML) rainfall observations with radar fields.
"""

from __future__ import annotations

__version__ = "0.0.0"

__all__ = ["__version__", "io", "radolan", "merge", "interpolate"]

from . import interpolate, io, merge, radolan
