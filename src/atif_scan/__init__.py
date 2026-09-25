"""Public Python API for composing ATIF review checks."""

from .checks import CheckSpec, Context, Detection, Detector, Severity, Status
from .detectors import (
    ObservationDetector,
    RegexDetector,
    SurfaceDetector,
    TraceCheck,
    builtin_detectors,
)
from .engine import Assessment, Engine
from .loader import TraceError, load_bytes, load_trace, parse_trace
from .model import Channel, Content, Locator, Observation, Step, Surface, ToolCall, Trace
from .report import document, report, to_json, to_text
from .rules import All, Allowance, AnyOf, Not, Ref, Requires, Rule

__all__ = [
    "All",
    "Allowance",
    "AnyOf",
    "Assessment",
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
    "ObservationDetector",
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
    "TraceCheck",
    "Trace",
    "TraceError",
    "builtin_detectors",
    "document",
    "load_bytes",
    "load_trace",
    "parse_trace",
    "report",
    "to_json",
    "to_text",
]
