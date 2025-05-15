# Copyright 2024-2025 LMCache Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional, no_type_check

from lmcache.experimental.memory_management import (MemoryAllocatorInterface,
                                                    MemoryObj)
# reuse
from lmcache.experimental.storage_backend.connector.base_connector import \
    RemoteConnector
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey

logger = init_logger(__name__)


class BlackholeConnector(RemoteConnector):

    def __init__(self, memory_allocator: MemoryAllocatorInterface):
        self.memory_allocator = memory_allocator

    async def exists(self, key: CacheEngineKey) -> bool:
        return False

    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        return None

    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        self.memory_allocator.ref_count_down(memory_obj)

    @no_type_check
    async def list(self) -> List[str]:
        pass

    async def close(self):
        logger.info("Closed the blackhole connection")
