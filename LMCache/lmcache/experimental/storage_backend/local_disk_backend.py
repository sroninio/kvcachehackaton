import asyncio
import os
import threading
import sys
from warnings import showwarning
sys.path.append('/workspace/external')

# ANSI escape codes for colors
BRIGHT_GREEN = "\033[92m"
BRIGHT_BLUE = "\033[94m"
BRIGHT_YELLOW = "\033[93m"
RESET = "\033[0m"
BRIGHT_RED = "\033[91m"

from collections import OrderedDict, defaultdict
from concurrent.futures import Future
from typing import TYPE_CHECKING, List, Optional
import global_vars 
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
from lmcache.logging import log_to_pid_file  

if TYPE_CHECKING:
    from lmcache.experimental.cache_controller.worker import LMCacheWorker

logger = init_logger(__name__)

BRIGHT_GREEN = "\033[92m"
BRIGHT_BLUE = "\033[94m"
BRIGHT_YELLOW = "\033[93m"
RESET = "\033[0m"

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

        log_to_pid_file(f"INIT LOCAL DISK BACKEND")
        self.prefetched = defaultdict(lambda: None) # key->mem_obj
        self.disk_lock = threading.Lock()
        assert config.local_disk is not None
        self.path: str = config.local_disk
        if not os.path.exists(self.path):
            log_to_pid_file("ERROR:BAD FILE")
        self.loop = loop
        self.memory_allocator = memory_allocator

    def __str__(self):
        return self.__class__.__name__

    def contains(self, key: CacheEngineKey) -> bool:
        with self.disk_lock:
            return key in self.prefetched

    def submit_put_task(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
    ) -> Optional[Future]:
        log_to_pid_file(f"IN SUBMIT PUT TASK key = {key}")
        assert memory_obj.tensor is not None
        global_vars.chunk_hashes_of_curr_batch.append(key)        
        self.memory_allocator.ref_count_up(memory_obj)
        future = asyncio.run_coroutine_threadsafe(
            self.async_save_bytes_to_disk(key, memory_obj), self.loop)
        return future
    

    async def prefetch_async(self, keys):
        print(f"{BRIGHT_YELLOW}LocalDiskBackend::prefetch called for keys {keys}{RESET}")
        self.disk_lock.acquire()
        tasks = []
        for key in keys:
            assert key in self.dict
            self.evictor.update_on_hit(key, self.dict)
            path = self.dict[key].path
            dtype = self.dict[key].dtype
            shape = self.dict[key].shape 
            assert dtype is not None
            assert shape is not None 
            tasks.append(asyncio.create_task(self.async_load_bytes_from_disk(path, dtype, shape)))
        self.disk_lock.release()
        return await asyncio.gather(*tasks)


    def add_to_prefetched(self, key, mem_obj):
        self.disk_lock.acquire()
        self.prefetched[key] = mem_obj
        self.disk_lock.release()

    
    def remove_from_prefteched(self, key):
        self.disk_lock.acquire()
        if key in self.prefetched:
            self.memory_allocator.free(self.prefetched[key]) 
            del self.prefetched[key] 
        self.disk_lock.release()


    def get_blocking(
        self,
        key: CacheEngineKey,
    ) -> Optional[MemoryObj]:
        """
        Blocking get function.
        """
        print(f"{BRIGHT_YELLOW}LocalDiskBackend::get_blocking called for key {key.to_string()}{RESET}")
        log_to_pid_file(f"IN GET BLOCKING key = {key}")
        with self.disk_lock:
            return self.prefetched[key]
    
    

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    async def async_save_bytes_to_disk(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
    ) -> None:
        """
        Convert KV to bytes and async store bytes to disk.
        """
        byte_array = memory_obj.byte_array
        async with aiofiles.open(self.path, 'r+b') as f:
            await f.seek(0)  # Move to offset 0
            await f.write(byte_array)
        self.memory_allocator.ref_count_down(memory_obj)


    # TODO(Jiayi): use `bytes_read = await f.readinto(buffer)`
    # for better performance (i.e., fewer copy)
    async def async_load_bytes_from_disk(
        self,
        path: str,
        dtype: torch.dtype,
        shape: torch.Size,
    ) -> Optional[MemoryObj]:
        """
        Async load bytearray from disk.
        """
        memory_obj = self.memory_allocator.allocate(shape, dtype)
        if memory_obj is None:
            logger.debug("Memory allocation failed during async disk load.")
            return None
        buffer = memory_obj.byte_array
        async with aiofiles.open(path, 'rb') as f:
            await f.readinto(buffer) 
        return memory_obj
