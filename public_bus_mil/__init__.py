"""public-bus-mil — reproducible exam-level MIL test set from public BUS videos.

See README.md for overview, DATASET_SPEC.md for the construction protocol.
"""

__version__ = "1.0.0"

from .frame_trim import trim_dark_margins
from .keyframe import select_keyframes, laplacian_variance
from .manifest import write_manifest, BAG_COLUMNS, FRAME_COLUMNS

__all__ = [
    "trim_dark_margins",
    "select_keyframes",
    "laplacian_variance",
    "write_manifest",
    "BAG_COLUMNS",
    "FRAME_COLUMNS",
]
