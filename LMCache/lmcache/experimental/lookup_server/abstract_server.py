import abc
from typing import List, Optional, Tuple

from lmcache.utils import CacheEngineKey


class LookupServerInterface(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def lookup(self, key: CacheEngineKey) -> Optional[Tuple[str, int]]:
        """
        Perform lookup in the lookup server.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def insert(
        self,
        key: CacheEngineKey,
    ):
        """
        Perform insert in the lookup server.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def remove(
        self,
        key: CacheEngineKey,
    ):
        """
        Perform remove in the lookup server.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def batched_remove(
        self,
        keys: List[CacheEngineKey],
    ):
        """
        Perform batched remove in the lookup server.
        """
        raise NotImplementedError