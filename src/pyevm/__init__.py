"""Eulerian Video Magnification — color, motion, and phase-based algorithms."""

from pyevm.magnification.color import ColorMagnifier
from pyevm.magnification.motion import MotionMagnifier
from pyevm.magnification.phase import PhaseMagnifier

__all__ = ["ColorMagnifier", "MotionMagnifier", "PhaseMagnifier"]
