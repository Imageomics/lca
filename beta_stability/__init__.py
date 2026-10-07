# -*- coding: utf-8 -*-
from beta_stability.util.init_logger import init_logger

init_logger()

try:
    from _version import __version__
except ImportError:
    __version__ = '0.0.0'

import beta_stability.util.cluster_tools  # NOQA
