"""Blindsight: cortical / retinal visual-prosthesis phosphene simulator."""

from .config import parse_config
from .simulator import Simulator

__all__ = ["Simulator", "parse_config"]
__version__ = "0.1.0"
