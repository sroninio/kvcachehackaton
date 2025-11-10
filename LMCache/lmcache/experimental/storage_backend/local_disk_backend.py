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
        #assert config.local_disk is not None
        #self.path: str = config.local_disk
        self.path: str = "/tmp/kvcache.bin"
        if not os.path.exists(self.path):
            log_to_pid_file("ERROR:BAD FILE")
            exit(1)
        self.loop = loop
        self.memory_allocator = memory_allocator
        self.dtype = None
        self.shape = None

    def __str__(self):
        return self.__class__.__name__

    def contains(self, key: CacheEngineKey) -> bool:
        with self.disk_lock:
            return True if self.prefetched[key] != None else False

    def submit_put_task(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
    ) -> Optional[Future]:
        log_to_pid_file(f"LDB IN SUBMIT PUT TASK key = {key}")
        assert memory_obj.tensor is not None
        to_store = True

        dtype, shape = memory_obj.metadata.dtype, memory_obj.metadata.shape
        if (self.dtype and (self.dtype != dtype)) or (self.shape and (self.shape != shape)):
            log_to_pid_file(f"LDB GOT DIFFERENT MEM OBJECTS DTYPE OR SHAPE")
            to_store = False
        else:
            self.dtype = dtype
            self.shape = shape
        global_vars.chunk_hashes_of_curr_batch.append(key)        
        self.memory_allocator.ref_count_up(memory_obj)
        future = asyncio.run_coroutine_threadsafe(
            self.async_save_bytes_to_disk(key, memory_obj, to_store), self.loop)
        return future
    

    async def prefetch_async(self, keys):
        log_to_pid_file(f"LDB IN PREFETCH ASYNC")
        self.disk_lock.acquire()
        tasks = []
        for key in keys:
            log_to_pid_file(f"LDB IN PREFETCH ASYNC key = {key}")
            tasks.append(asyncio.create_task(self.async_load_bytes_from_disk()))
        self.disk_lock.release()
        return await asyncio.gather(*tasks)


    def add_to_prefetched(self, key, mem_obj):
        self.disk_lock.acquire()
        log_to_pid_file(f"LDB adding to prefetched key = {key}")
        self.prefetched[key] = mem_obj
        self.disk_lock.release()

    
    def remove_from_prefteched(self, key):
        self.disk_lock.acquire()
        if key in self.prefetched:
            self.memory_allocator.free(self.prefetched[key]) 
            log_to_pid_file(f"LDB removing from prefetched key = {key}")
            del self.prefetched[key] 
        self.disk_lock.release()


    def get_blocking(
        self,
        key: CacheEngineKey,
    ) -> Optional[MemoryObj]:
        """
        Blocking get function.
        """
        log_to_pid_file(f"LDB IN GET BLOCKING key = {key}")
        with self.disk_lock:
            ret = self.prefetched[key]
            log_to_pid_file(f"LDB GET BLOCKING FOUND = {ret != None}")
            return ret

    
    

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    async def async_save_bytes_to_disk(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
        to_store: bool
    ) -> None:
        """
        Convert KV to bytes and async store bytes to disk.
        """
        if to_store:
            byte_array = memory_obj.byte_array
            async with aiofiles.open(self.path, 'r+b') as f:
                await f.seek(0)  # Move to offset 0
                await f.write(byte_array)
        self.memory_allocator.ref_count_down(memory_obj)


    # TODO(Jiayi): use `bytes_read = await f.readinto(buffer)`
    # for better performance (i.e., fewer copy)
    async def async_load_bytes_from_disk(self) -> Optional[MemoryObj]:
        """
        Async load bytearray from disk.
        """
        if (not self.shape) or (not self.dtype):
            log_to_pid_file(f"LDB ASYNC LOADING WITHOUT SHAPES")
            exit(1)

        memory_obj = self.memory_allocator.allocate(self.shape, self.dtype)
        if memory_obj is None:
            log_to_pid_file(f"Memory allocation failed during async disk load.")
            return None
        buffer = memory_obj.byte_array
        async with aiofiles.open(self.path, 'rb') as f:
            await f.readinto(buffer) 
        return memory_obj
    
    def close(self):
        return

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        log_to_pid_file(f"LDB CHECKS IF EXISTS IN PUT TASKS = {key}")
        return False

    def submit_prefetch_task(
        self,
        key: CacheEngineKey,
    ) -> Optional[Future]:
        log_to_pid_file(f"LDB IN SUBMITTING PREFETCH TASK (SHOULDNT BE HERE)= {key}")
        return None


