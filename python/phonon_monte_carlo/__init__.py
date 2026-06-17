"""
Python bindings to the program phonon_monte_carlo.exe

Example
------
```
    from phonon_monte_carlo import run
    run("/path/to/config.toml", "/path/to/sources.parquet")
```
"""

from .phonon_monte_carlo import run

__all__ = ["run"]
