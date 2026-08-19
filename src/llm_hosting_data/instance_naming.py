"""Shared AWS instance-identifier parsing.

Used by both the live pricing fetch (``aws_pricing.py``) and the static GPU
reference table (``gpu_reference.py``) — split out to a dependency-free
module so those two don't need to import each other.
"""

from __future__ import annotations


def family_of(instance_identifier: str) -> str:
    """Return the family token AWS's own `<family>.<size>` naming uses.

    E.g. ``"g6e.2xlarge"`` -> ``"g6e"``, ``"ml.g7e.2xlarge"`` -> ``"g7e"``.
    Splitting on the first "." (after stripping SageMaker's "ml." prefix)
    keeps ``p4d`` and ``p4de`` distinct, which a plain string-prefix match
    would not: ``"p4de.24xlarge".startswith("p4d")`` is true, so naive
    prefix matching would silently pull p4de instances into a "p4d" family.
    """
    identifier = instance_identifier.removeprefix("ml.")
    return identifier.split(".", 1)[0]
