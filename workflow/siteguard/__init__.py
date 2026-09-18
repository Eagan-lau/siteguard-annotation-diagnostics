"""SiteGuard V4 public package."""

from .decision import LEVELS, decide_highest_supported_resolution

__all__ = ["LEVELS", "decide_highest_supported_resolution"]
__version__ = "4.2.0.dev0"
