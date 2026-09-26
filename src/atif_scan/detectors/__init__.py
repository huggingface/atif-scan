from .builtin import builtin_detectors
from .integrity import TraceCheck, integrity_detectors
from .text import ObservationDetector, OwnTaskFiles, RegexDetector, SurfaceDetector

__all__ = [
    "ObservationDetector",
    "OwnTaskFiles",
    "RegexDetector",
    "SurfaceDetector",
    "TraceCheck",
    "builtin_detectors",
    "integrity_detectors",
]
