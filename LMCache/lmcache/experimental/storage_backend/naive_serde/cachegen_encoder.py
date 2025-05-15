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

import torch

from lmcache.config import LMCacheEngineMetadata
from lmcache.experimental.config import LMCacheEngineConfig
from lmcache.experimental.memory_management import (BytesBufferMemoryObj,
                                                    MemoryAllocatorInterface,
                                                    MemoryObj)
from lmcache.experimental.storage_backend.naive_serde.cachegen_basics import \
    CacheGenConfig
from lmcache.experimental.storage_backend.naive_serde.serde import Serializer
from lmcache.logging import init_logger
from lmcache.storage_backend.serde.cachegen_encoder import encode_function
from lmcache.utils import _lmcache_nvtx_annotate

logger = init_logger(__name__)


class CacheGenSerializer(Serializer):

    def __init__(self, config: LMCacheEngineConfig,
                 metadata: LMCacheEngineMetadata,
                 memory_allocator: MemoryAllocatorInterface):
        self.cachegen_config = CacheGenConfig.from_model_name(
            metadata.model_name)
        self.chunk_size = config.chunk_size
        self.fmt = metadata.fmt
        self.key_bins = self.make_key_bins(self.cachegen_config)
        self.value_bins = self.make_value_bins(self.cachegen_config)

        self.kv_shape = metadata.kv_shape

        self.memory_allocator = memory_allocator

    def make_key_bins(self, config: CacheGenConfig) -> torch.Tensor:
        ret = torch.zeros(config.nlayers)
        for spec in config.kspecs:
            ret[spec.start_layer:spec.end_layer] = spec.bins
        return ret.cuda()

    def make_value_bins(self, config: CacheGenConfig) -> torch.Tensor:
        ret = torch.zeros(config.nlayers)
        for spec in config.vspecs:
            ret[spec.start_layer:spec.end_layer] = spec.bins
        return ret.cuda()

    # TODO(Jiayi): A lot of memory copies can be avoided in this function.
    @_lmcache_nvtx_annotate
    def serialize(self, memory_obj: MemoryObj) -> BytesBufferMemoryObj:
        """
        Serialize a KV_BLOB MemoryObj to CACHEGEN_BINARY MemoryObj.

        Input:
            memory_obj: the memory object to be serialized.

        Returns:
            MemoryObj: the serialized binary memory object.
        """

        # TODO(Jiayi): please avoid this copy by directly performing
        # serialization inside gpu connector.
        assert memory_obj.tensor is not None
        tensor = memory_obj.tensor.cuda()

        # Temporary fix for issue #83: encoder will have the default device 0
        # on all the ray workers. Need to set it to the correct device.
        # Also need to figure out why this happens.
        if torch.cuda.current_device != tensor.device:
            torch.cuda.set_device(tensor.device)
        if tensor.device != self.key_bins.device:
            self.key_bins = self.key_bins.to(tensor.device)
        if tensor.device != self.value_bins.device:
            self.value_bins = self.value_bins.to(tensor.device)

        # tensor is [2, num_layers, num_tokens, hidden_size]
        tensor = tensor.view(*tensor.shape[:-1], self.kv_shape[-2],
                             self.kv_shape[-1])
        tensor = tensor.permute([1, 0, 2, 3, 4])

        # TODO(Jiayi): remove hardcoded "2"
        """ expecting a tensor of shape 
        [num_layers, 2, num_tokens, num_heads, head_size] """
        ntokens = tensor.shape[2]
        output_dict = encode_function(
            tensor,
            self.cachegen_config,
            self.key_bins,
            self.value_bins,
            ntokens,
        )

        return BytesBufferMemoryObj(output_dict.to_bytes())
