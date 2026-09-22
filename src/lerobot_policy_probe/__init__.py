"""Find out whether a LeRobot policy is actually trained, before you put it on hardware.

A fine-tune can fail silently: a component that never loaded is randomly initialised, and
`from_pretrained` swallows the error. The model trains, saves, uploads and deploys, and is
simply blind. These tools catch that, and separate model faults from transport faults.
"""

from .inspect_weights import Finding, inspect, looks_random
from .score import ScoreResult, reversals, score

__version__ = "0.1.0"

__all__ = [
    "Finding",
    "ScoreResult",
    "__version__",
    "inspect",
    "looks_random",
    "reversals",
    "score",
]
