"""Bayesian spacecraft-acquisition search and antenna pointing."""

from importlib.metadata import PackageNotFoundError, version

from antenna_pomdp.config import Config

try:
    __version__ = version("leopt")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = ["Config", "__version__"]
