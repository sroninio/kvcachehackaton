import asyncio
from collections import OrderedDict
from typing import TYPE_CHECKING, Optional
import sys
sys.path.append('/workspace/external')
import global_vars

import torch

from lmcache.config import LMCacheEngineMetadata
from lmcache.experimental.config import LMCacheEngineConfig
from lmcache.experimental.lookup_server import LookupServerInterface
from lmcache.experimental.memory_management import MemoryAllocatorInterface
from lmcache.experimental.storage_backend.abstract_backend import \
    StorageBackendInterface
from lmcache.experimental.storage_backend.local_disk_backend import \
    LocalDiskBackend
from lmcache.experimental.storage_backend.remote_backend import RemoteBackend
from lmcache.logging import init_logger

from lmcache.experimental.storage_backend.local_disk_backend_rai import LocalDiskBackend as LocalDiskBackend_rai

if TYPE_CHECKING:
    from lmcache.experimental.cache_controller.worker import LMCacheWorker


logger = init_logger(__name__)


def CreateStorageBackends(
    config: LMCacheEngineConfig,
    metadata: LMCacheEngineMetadata,
    loop: asyncio.AbstractEventLoop,
    memory_allocator: MemoryAllocatorInterface,
    dst_device: str = "cuda",
    lmcache_worker: Optional["LMCacheWorker"] = None,
    lookup_server: Optional[LookupServerInterface] = None,
) -> OrderedDict[str, StorageBackendInterface]:

    # Replace 'cuda' with 'cuda:<device id>'
    if dst_device == "cuda":
        dst_device = f"cuda:{torch.cuda.current_device()}"

    storage_backends: OrderedDict[str, StorageBackendInterface] =\
        OrderedDict()

    if config.is_rai:
        local_disk_backend = LocalDiskBackend_rai(config, loop, memory_allocator,
                                              dst_device, lmcache_worker,
                                              lookup_server)
        backend_name = str(local_disk_backend)        
        storage_backends[backend_name] = local_disk_backend        

    # TODO(Jiayi): The hierarchy is fixed for now
    elif config.local_disk and config.max_local_disk_size > 0:
        local_disk_backend = LocalDiskBackend(config, loop, memory_allocator,
                                              dst_device, lmcache_worker,
                                              lookup_server)
        backend_name = str(local_disk_backend)
        print(f"CREATING STORAGE BACKEND")
        global_vars.backend = local_disk_backend
        storage_backends[backend_name] = local_disk_backend

    if config.remote_url is not None:
        remote_backend = RemoteBackend(config, metadata, loop,
                                       memory_allocator, dst_device,
                                       lookup_server)
        backend_name = str(remote_backend)
        storage_backends[backend_name] = remote_backend

    # TODO(Jiayi): Please support other backends
    config.enable_blending = False
    assert config.enable_blending is False, \
        "blending is not supported for now"

    return storage_backends
