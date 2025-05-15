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

import asyncio
import ctypes
import operator
from functools import reduce
from typing import List, Optional, Union, no_type_check

import infinistore
import torch

from lmcache.experimental.memory_management import (CopyLessMemoryObj,
                                                    MemoryAllocatorInterface,
                                                    MemoryObj)
# reuse
from lmcache.experimental.protocol import RedisMetadata
from lmcache.experimental.storage_backend.connector.base_connector import \
    RemoteConnector
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey

logger = init_logger(__name__)

MAX_BUFFER_SIZE = 40 << 20  # 40MB
METADATA_BYTES_LEN = 28
MAX_BUFFER_CNT = 16


def _get_ptr(mv: Union[bytearray, memoryview]) -> int:
    return ctypes.addressof(ctypes.c_char.from_buffer(mv))


class InfinistoreConnector(RemoteConnector):

    def __init__(self, host: str, port: int, dev_name,
                 loop: asyncio.AbstractEventLoop,
                 memory_allocator: MemoryAllocatorInterface):
        config = infinistore.ClientConfig(
            host_addr=host,
            service_port=port,
            log_level="info",
            connection_type=infinistore.TYPE_RDMA,
            ib_port=1,
            link_type=infinistore.LINK_ETHERNET,
            dev_name=dev_name,
        )

        self.rdma_conn = infinistore.InfinityConnection(config)

        self.loop = loop
        self.rdma_conn.connect()

        self.send_buffers = []
        self.recv_buffers = []
        self.send_queue: asyncio.Queue[int] = asyncio.Queue(
            maxsize=MAX_BUFFER_CNT)
        self.recv_queue: asyncio.Queue[int] = asyncio.Queue(
            maxsize=MAX_BUFFER_CNT)

        self.buffer_size = MAX_BUFFER_SIZE
        for i in range(MAX_BUFFER_CNT):
            send_buffer = bytearray(self.buffer_size)
            self.rdma_conn.register_mr(_get_ptr(send_buffer), self.buffer_size)
            self.send_buffers.append(send_buffer)
            self.send_queue.put_nowait(i)

            recv_buffer = bytearray(self.buffer_size)
            self.rdma_conn.register_mr(_get_ptr(recv_buffer), self.buffer_size)
            self.recv_buffers.append(recv_buffer)
            self.recv_queue.put_nowait(i)

    async def exists(self, key: CacheEngineKey) -> bool:

        def blocking_io():
            return self.rdma_conn.check_exist(key.to_string())

        return await self.loop.run_in_executor(None, blocking_io)

    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        key_str = key.to_string()

        buf_idx = await self.recv_queue.get()
        buffer = self.recv_buffers[buf_idx]
        try:
            await self.rdma_conn.rdma_read_cache_async([(key_str, 0)],
                                                       self.buffer_size,
                                                       _get_ptr(buffer))
        except Exception as e:
            logger.warning(f"get failed: {e}")
            self.recv_queue.put_nowait(buf_idx)
            return None

        metadata = RedisMetadata.deserialize(buffer)

        def callback():
            self.recv_queue.put_nowait(buf_idx)

        num_elements = reduce(operator.mul, metadata.shape)
        assert metadata.dtype is not None
        temp_tensor = torch.frombuffer(buffer,
                                       dtype=metadata.dtype,
                                       offset=METADATA_BYTES_LEN,
                                       count=num_elements).reshape(
                                           metadata.shape)

        memory_obj = CopyLessMemoryObj(raw_data=temp_tensor,
                                       metadata=metadata,
                                       callback=callback)

        logger.debug(f"get key: {key_str} done, {memory_obj.get_shape()}")
        return memory_obj

    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj):

        key_str = key.to_string()

        kv_bytes = memory_obj.byte_array
        kv_shape = memory_obj.get_shape()
        kv_dtype = memory_obj.get_dtype()
        memory_format = memory_obj.get_memory_format()

        buf_idx = await self.send_queue.get()
        buffer = self.send_buffers[buf_idx]

        RedisMetadata(len(kv_bytes), kv_shape, kv_dtype,
                      memory_format).serialize_into(buffer)

        buffer[METADATA_BYTES_LEN:METADATA_BYTES_LEN +
               len(kv_bytes)] = kv_bytes

        size = memory_obj.get_size()

        if size + METADATA_BYTES_LEN > self.buffer_size:
            raise ValueError(
                f"Value size ({size + METADATA_BYTES_LEN} bytes)"
                f"exceeds the maximum allowed size"
                f"({self.buffer_size} bytes). Please decrease chunk_size.")
        try:
            await self.rdma_conn.rdma_write_cache_async(
                [(key_str, 0)], METADATA_BYTES_LEN + size, _get_ptr(buffer))
        except Exception as e:
            logger.warning(
                f"exception happens in rdma_write_cache_async kv_bytes {e}")
            return
        finally:
            self.send_queue.put_nowait(buf_idx)

        logger.debug(f"put key: {key.to_string()}, {memory_obj.get_shape()}")

    # TODO
    @no_type_check
    async def list(self) -> List[str]:
        pass

    async def close(self):
        self.rdma_conn.close()
        logger.info("Closed the infinistore connection")
