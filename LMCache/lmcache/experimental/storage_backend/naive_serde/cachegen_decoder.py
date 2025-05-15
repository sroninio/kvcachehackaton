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

from typing import Optional

import torch

from lmcache.config import LMCacheEngineMetadata
from lmcache.experimental.config import LMCacheEngineConfig
from lmcache.experimental.memory_management import (BytesBufferMemoryObj,
                                                    MemoryAllocatorInterface,
                                                    MemoryFormat, MemoryObj,
                                                    MemoryObjMetadata,
                                                    TensorMemoryObj)
from lmcache.experimental.storage_backend.naive_serde.cachegen_basics import \
    CacheGenConfig
from lmcache.experimental.storage_backend.naive_serde.serde import Deserializer
from lmcache.logging import init_logger
from lmcache.storage_backend.serde.cachegen_basics import \
    CacheGenGPUEncoderOutput
from lmcache.storage_backend.serde.cachegen_decoder import (
    decode_function_gpu, do_dequantize)
from lmcache.utils import _lmcache_nvtx_annotate

logger = init_logger(__name__)


class CacheGenDeserializer(Deserializer):

    def __init__(self, config: LMCacheEngineConfig,
                 metadata: LMCacheEngineMetadata,
                 memory_allocator: MemoryAllocatorInterface):
        self.dtype = metadata.kv_dtype
        self.cachegen_config = CacheGenConfig.from_model_name(
            metadata.model_name)
        self.chunk_size = config.chunk_size
        self.output_buffer: Optional[torch.Tensor] = None
        self.fmt = metadata.fmt
        self.key_bins = self.make_key_bins(self.cachegen_config)
        self.value_bins = self.make_value_bins(self.cachegen_config)

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

    def get_output_buffer(self, nlayers: int, nchannels: int, ntokens: int):
        if (self.output_buffer is None
                or self.output_buffer.shape[1] != 2 * nlayers * nchannels):
            self.output_buffer = torch.zeros(
                (self.chunk_size, 2 * nlayers * nchannels),
                dtype=torch.uint8).cuda()
        return self.output_buffer[:ntokens, :]

    # TODO(Jiayi): A lot of memory copies can be avoided in this function.
    @_lmcache_nvtx_annotate
    def deserialize(
            self,
            buffer_memory_obj: BytesBufferMemoryObj) -> Optional[MemoryObj]:
        encoder_output = CacheGenGPUEncoderOutput.from_bytes(
            buffer_memory_obj.byte_array)

        encoder_output.max_tensors_key = encoder_output.max_tensors_key.cuda()
        encoder_output.max_tensors_value = (
            encoder_output.max_tensors_value.cuda())

        ntokens = encoder_output.max_tensors_key.shape[1]
        layers_in_key = encoder_output.max_tensors_key.shape[0]
        key, value = decode_function_gpu(
            encoder_output.cdf,
            encoder_output.data_chunks,
            layers_in_key,
            ntokens,
            self.get_output_buffer(
                encoder_output.cdf.shape[0] // 2,
                encoder_output.cdf.shape[1],
                ntokens,
            ),
        )

        # Temporary fix for #83: change the device of key_bins and value_bins
        # to the device of key and value
        # This requires a long-term fix in the future. Currently,
        # CacheGenGPUEncoderOutput has implicit device in itself.
        # More specifically, if the encoder encodes the tensor on GPU0, the
        # from_bytes will also return a tensor on GPU0
        # We may want to dynamically configure the device based on config and
        # metadata in the future
        if self.key_bins.device != key.device:
            self.key_bins = self.key_bins.to(key.device)

        if self.value_bins.device != value.device:
            self.value_bins = self.value_bins.cuda()

        key = do_dequantize(key, self.key_bins, encoder_output.max_tensors_key)
        value = do_dequantize(value, self.value_bins,
                              encoder_output.max_tensors_value)
        """ merge key and value back and reshape """
        nlayers, ntokens, nchannels = key.shape
        blob = torch.stack([key, value])  # [2, nlayers, ntokens, nchannels]
        blob = blob.reshape((
            2,
            nlayers,
            ntokens,
            encoder_output.num_heads,
            encoder_output.head_size,
        ))
        match self.fmt:
            case "vllm":
                hidden_dim = blob.shape[-1] * blob.shape[-2]
                kv_chunk = blob.reshape(*blob.shape[:-2], hidden_dim).to(
                    self.dtype)  # [nlayers, 2, ntokens, num_heads, head_size]
            case _:
                raise RuntimeError("Unknown format %s" % self.fmt)

        memory_obj = TensorMemoryObj(
            raw_data=kv_chunk,
            metadata=MemoryObjMetadata(
                shape=kv_chunk.shape,
                dtype=kv_chunk.dtype,
                address=-1,
                phy_size=kv_chunk.numel() * kv_chunk.element_size(),
                ref_count=-1,  # HACK: avoid mis-free
                fmt=MemoryFormat.KV_BLOB))

        return memory_obj

        #memory_obj = self.memory_allocator.allocate(kv_chunk.shape,
        #                                            kv_chunk.dtype,
        #                                            fmt=MemoryFormat.KV_BLOB)

        #if memory_obj is None:
        #    logger.warning("Memory allocation failed in cachegen deserializer")
        #    return None

        #assert memory_obj.tensor is not None
        #memory_obj.tensor.copy_(kv_chunk)

        #return memory_obj
