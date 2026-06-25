"""Compatibility bridge for the legacy tracking engine module.

New code should import from :mod:`pig_behavior.tracking`. This module exists so
older notebooks and scripts that still import
``pig_behavior.data_preparation.tracking_engine`` keep working without carrying
the old monolithic implementation.
"""

# ruff: noqa: I001

from __future__ import annotations

from pig_behavior.tracking import *  # noqa: F403
from pig_behavior.tracking import __all__ as _tracking_all, cli as _cli

for _name in _cli.__all__:
    globals()[_name] = getattr(_cli, _name)

__all__ = [*_tracking_all, *_cli.__all__]


if __name__ == "__main__":
    raise SystemExit(_cli.main())
