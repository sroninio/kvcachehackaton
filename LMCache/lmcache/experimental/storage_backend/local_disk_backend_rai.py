import asyncio
import os
import threading
from collections import OrderedDict
from concurrent.futures import Future
from typing import TYPE_CHECKING, List, Optional

import aiofiles
import torch

from lmcache.experimental.cache_controller.message import (KVAdmitMsg,
                                                           KVEvictMsg)
from lmcache.experimental.config import LMCacheEngineConfig
from lmcache.experimental.lookup_server import LookupServerInterface
from lmcache.experimental.memory_management import (MemoryAllocatorInterface,
                                                    MemoryObj)
from lmcache.experimental.storage_backend.abstract_backend import \
    StorageBackendInterface
from lmcache.experimental.storage_backend.evictor import LRUEvictor, PutStatus
from lmcache.logging import init_logger
from lmcache.observability import LMCStatsMonitor
from lmcache.utils import (CacheEngineKey, DiskCacheMetadata,
                           _lmcache_nvtx_annotate)

if TYPE_CHECKING:
    from lmcache.experimental.cache_controller.worker import LMCacheWorker

logger = init_logger(__name__)


class LocalDiskBackend(StorageBackendInterface):

    def __init__(
        self,
        config: LMCacheEngineConfig,
        loop: asyncio.AbstractEventLoop,
        memory_allocator: MemoryAllocatorInterface,
        dst_device: str = "cuda",
        lmcache_worker: Optional["LMCacheWorker"] = None,
        lookup_server: Optional[LookupServerInterface] = None,
    ):
        print("RAI INIT")


    def __str__(self):
        return "KARAMBA"

    def _key_to_path(
        self,
        key: CacheEngineKey,
    ) -> str:
        return "/x/y/z.pt"

    def contains(self, key: CacheEngineKey) -> bool:
        print("RAI CONTAINS")
        return False

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        print("RAI EXISTS")
        return False

    def remove(
        self,
        key: CacheEngineKey,
    ) -> None:
        print("RAI REMOVE")

    def insert_key(self, key: CacheEngineKey, memory_obj: MemoryObj) -> None:
        print("RAI INSERT")


    def submit_put_task(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
    ) -> Optional[Future]:
        print("RAI SUBMIT PUT TASK")
        return None

    def submit_prefetch_task(
        self,
        key: CacheEngineKey,
    ) -> Optional[Future]:
        print("RAI SUBMIT PREFETCH TASK")
        return None

    def get_blocking(
        self,
        key: CacheEngineKey,
    ) -> Optional[MemoryObj]:
        """
        Blocking get function.
        """
        print("RAI SUBMIT GET BLOCKING TASK")
        return None


    def close(self) -> None:
        print("RAI CLOSE")

