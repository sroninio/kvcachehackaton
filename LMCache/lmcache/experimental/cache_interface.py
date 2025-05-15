from typing import Optional

import msgspec


class LMCacheModelRequest(
        msgspec.Struct,
        array_like=True,  # type: ignore[call-arg]
        omit_defaults=True):  # type: ignore[call-arg]
    """
    User-provided information to control the cache behavior.
    """
    store_cache: bool = True  # Whether to store the cache
    ttl: Optional[float] = None  # Time to live
