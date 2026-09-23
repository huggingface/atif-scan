"""Public Python API for composing ATIF review checks."""

from .checks import CheckSpec, Context, Detection, Detector, Severity, Status
from .detectors import RegexDetector, SurfaceDetector, builtin_detectors
from .engine import Engine, report
from .loader import TraceError, load_trace, parse_trace
from .model import Channel, Content, Locator, Observation, Step, Surface, ToolCall, Trace
from .rules import All, AnyOf, Not, Ref, Requires, Rule

__all__ = [
    "All",
    "AnyOf",
    "Channel",
    "CheckSpec",
    "Content",
    "Context",
    "Detection",
    "Detector",
    "Engine",
    "Locator",
    "Observation",
    "Not",
    "Ref",
    "RegexDetector",
    "Requires",
    "Rule",
    "Severity",
    "Status",
    "Step",
    "Surface",
    "SurfaceDetector",
    "ToolCall",
    "Trace",
    "TraceError",
    "builtin_detectors",
    "load_trace",
    "parse_trace",
    "report",
]
