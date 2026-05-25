import math
from collections import Counter

from .common import flatten
from .fisher_yates import fisher_yates


def corgi_pile(data, B=3):
    """The one-round CorgiPile baseline used in the paper comparison."""
    if not data:
        return []
    blocks = [
        data[i * B:(i + 1) * B]
        for i in range(math.ceil(len(data) / B))
    ]
    shuffled_blocks = fisher_yates(blocks)
    gates = [
        fisher_yates(flatten(shuffled_blocks[i * B:(i + 1) * B]))
        for i in range(math.ceil(len(data) / (B ** 2)))
    ]
    result = flatten(gates)
    if __debug__:
        assert len(result) == len(data)
        assert Counter(result) == Counter(data)
    return result
