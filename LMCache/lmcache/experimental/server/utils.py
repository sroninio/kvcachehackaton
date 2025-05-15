from dataclasses import dataclass
from typing import Optional

import torch

from lmcache.experimental.memory_management import MemoryFormat


# TODO(Jiayi): Maybe move the memory management in remote
# cache server to `memory_management.py` as well.
@dataclass
class LMSMemoryObj:
    data: bytearray
    length: int
    fmt: MemoryFormat
    dtype: Optional[torch.dtype]
    shape: torch.Size
