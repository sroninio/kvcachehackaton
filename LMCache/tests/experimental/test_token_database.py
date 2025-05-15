import hashlib

import pytest
import torch
from utils import dumb_metadata, dumb_metadata_with_model_name, generate_tokens

from lmcache.experimental.config import LMCacheEngineConfig
from lmcache.experimental.token_database import (ChunkedTokenDatabase,
                                                 SegmentTokenDatabase)


@pytest.mark.parametrize('chunk_length', [16, 64, 256])
def test_chunked_token_database(chunk_length):
    cfg = LMCacheEngineConfig.from_legacy(chunk_size=chunk_length,
                                          backend="cpu")
    metadata = dumb_metadata()

    test_length = 2500
    tokens = generate_tokens(test_length, "cpu")
    mask = torch.full([test_length], True, dtype=torch.bool, device="cpu")

    num_falses = [
        i * chunk_length for i in range(0, test_length // chunk_length)
    ]

    db = ChunkedTokenDatabase(cfg, metadata)

    # Process without mask
    original_results = list(db.process_tokens(tokens))
    for i in range(0, test_length, chunk_length):
        st, ed, key = original_results[i // chunk_length]
        assert st == i
        assert ed == min(i + chunk_length, test_length)

    for i in range(0, test_length // chunk_length):
        mask[:num_falses[i]] = False
        new_results = list(db.process_tokens(tokens, mask))
        assert len(new_results) == len(original_results) - i

        for j in range(len(new_results)):
            st, ed, key = new_results[j]
            assert st == original_results[j + i][0]
            assert ed == original_results[j + i][1]


@pytest.mark.parametrize('prefix_length', [0, 16, 64, 256])
@pytest.mark.parametrize('chunk_lengths', [[256, 512, 256], [1024, 512, 256]])
def test_segment_token_database(prefix_length, chunk_lengths):
    cfg = LMCacheEngineConfig.from_legacy(blend_special_str=" # # ")
    metadata = dumb_metadata_with_model_name("facebook/opt-125m")

    db = SegmentTokenDatabase(cfg, metadata)
    sep_tokens = db.sep_tokens

    sys_length = 25
    query_length = 50
    sys_tokens = generate_tokens(sys_length, "cpu", fixed=True)
    query_tokens = generate_tokens(query_length, "cpu", fixed=True)

    token_chunks = []
    starts = [0]
    ends = [sys_length]
    sys_bytes = sys_tokens.cpu().to(torch.uint32).numpy().tobytes()
    sys_hash = hashlib.sha256(sys_bytes).hexdigest()
    hashes = [sys_hash]
    start = sys_length + len(sep_tokens)
    for idx, chunk_length in enumerate(chunk_lengths):
        token_chunk = generate_tokens(chunk_length, "cpu", fixed=True)

        token_bytes = token_chunk.cpu().to(torch.uint32).numpy().tobytes()
        token_hash = hashlib.sha256(token_bytes).hexdigest()
        hashes.append(token_hash)

        token_chunk = torch.cat([sep_tokens, token_chunk])
        token_chunks.append(token_chunk)
        starts.append(start)
        ends.append(start + chunk_length)
        start += chunk_length + len(sep_tokens)

    query_bytes = query_tokens.cpu().to(torch.uint32).numpy().tobytes()
    query_hash = hashlib.sha256(query_bytes).hexdigest()
    hashes.append(query_hash)

    tokens = torch.cat([sys_tokens, *token_chunks, sep_tokens, query_tokens])
    total_length = len(tokens)
    mask = torch.full([total_length], True, dtype=torch.bool, device="cpu")
    mask[:prefix_length] = False

    chunk_lists = [sys_tokens, *token_chunks, sep_tokens, query_tokens]
    skip_chunk_num = 0
    cum_length = 0
    for chunk in chunk_lists:
        if prefix_length > cum_length:
            skip_chunk_num += 1
        cum_length += len(chunk)

    starts = starts[skip_chunk_num:]
    ends = ends[skip_chunk_num:]
    hashes = hashes[skip_chunk_num:]

    original_results = list(db.process_tokens(tokens, mask))
    for i in range(len(original_results)):
        st, ed, key = original_results[i]
        assert st == starts[i]
        assert ed == ends[i]
        assert key.chunk_hash == hashes[i]
        #print(st, starts[i])
        #print(ed, ends[i])
