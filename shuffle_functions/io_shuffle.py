from collections import Counter

from .common import uniform_shuffle


def io_shuffle(data, B=3, rounds=2):
    """Implementation of Algorithm 2, IO Shuffle."""
    N = len(data)
    if N == 0:
        return []
    if __debug__:
        original_counts = Counter(data)
    block_size = B
    gate_block_count = B
    for _ in range(rounds):
        blocks = [
            uniform_shuffle(data[i:i + block_size])
            for i in range(0, N, block_size)
        ]
        blocks = uniform_shuffle(blocks)
        output = []
        for g in range(0, len(blocks), gate_block_count):
            gate_blocks = blocks[g:g + gate_block_count]
            if not gate_blocks:
                continue
            gate_lens = [len(block) for block in gate_blocks]
            max_len = max(gate_lens)
            for i in range(max_len):
                for block_idx, block in enumerate(gate_blocks):
                    if i < gate_lens[block_idx]:
                        output.append(block[i])
        data = output
        if __debug__:
            assert len(data) == N
    if __debug__:
        assert Counter(data) == original_counts
    return data
