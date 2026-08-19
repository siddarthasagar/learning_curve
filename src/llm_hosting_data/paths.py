"""Shared local-storage roots for this pipeline's cached source data and output.

Both live under ``dump/`` (relative to the working directory), so everything
this pipeline writes to disk — regenerable snapshots and raw source-data
caches alike — is in one place: inspectable, and clearable in one shot with
``rm -rf dump/``. Kept in its own dependency-free module since ``cli.py``,
``aws_pricing.py``, and ``gpu_reference.py`` all need it and none of those
should have to import each other just to agree on where the cache lives.
"""

from __future__ import annotations

from pathlib import Path

DUMP_DIR = Path("dump")

# Raw/extracted *source* caches (AWS offer files, the Vantage catalog) live
# here — one copy each, overwritten in place, deliberately not versioned
# like the timestamped output snapshots directly under DUMP_DIR.
CACHE_DIR = DUMP_DIR / "cache"
