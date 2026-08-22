"""
Stand-in for a package that several generators import.

Whoever imports it adds itself to :data:`users`, so a generator can tell how many
generators before it had already paid for this import.
"""

from __future__ import annotations

from typing_extensions import List

users: List[str] = []
"""
Names of the packages whose generator has imported this module, in the order they did.
"""
