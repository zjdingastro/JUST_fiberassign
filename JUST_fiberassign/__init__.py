"""
JUST_fiberassign: Fiber Assignment Package for JUST Telescope

This package provides algorithms for assigning spectroscopic fibers
to astronomical targets for the Jiao Tong University Spectroscopic Telescope (JUST).
"""

from .assign import assign_targets_greedy
from .utils import *
from .fiber_motion import *

__version__ = "0.1.0"
__author__ = "JUST Telescope Team"
__description__ = "Fiber assignment algorithms for JUST telescope"