from __future__ import annotations

from collections.abc import Mapping
from typing import TypeAlias

import h5py
import npc_io

StimPathOrDataset: TypeAlias = npc_io.PathLike | h5py.File | Mapping
"""Type alias for a path to a stim file, an open h5py file or a dict loaded
from a pickle file"""
