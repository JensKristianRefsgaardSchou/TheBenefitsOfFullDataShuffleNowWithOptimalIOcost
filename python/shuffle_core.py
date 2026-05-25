import math
import random
from collections import Counter


def uniform_shuffle(data):
    return random.sample(data, len(data))


def flatten(rows):
    result = []
    for row in rows:
        result.extend(row)
    return result


def fisher_yates(data, B=None):
    result = list(data)
    for i in range(len(result) - 1, 0, -1):
        j = random.randint(0, i)
        result[i], result[j] = result[j], result[i]
    return result


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


def mod_inverse(a, n):
    def extended_gcd(x, y):
        if x == 0:
            return y, 0, 1
        gcd, x1, y1 = extended_gcd(y % x, x)
        return gcd, y1 - (y // x) * x1, x1

    gcd, x, _ = extended_gcd(a % n, n)
    if gcd != 1:
        raise ValueError(f"Modular inverse does not exist for {a} mod {n}")
    return x % n


def gen_2_wise_ind_perm(C, B=3):
    """Implementation of Algorithm 1, Gen-2-Wise-Ind-Perm."""
    N = len(C)
    if N <= 1:
        return list(C)

    while True:
        a = random.randint(1, N - 1)
        if math.gcd(a, N) == 1:
            break
    a_inv = mod_inverse(a, N)

    pairs = []
    for b1 in range(-B + 1, B):
        for b2 in range(-B + 1, B):
            s = a * b1 + b2
            s_prime = b1 + a_inv * b2
            pairs.append((s, s_prime))

    output = [None] * N
    seen = [False] * N
    remaining = N
    while remaining:
        i = next(idx for idx, is_seen in enumerate(seen) if not is_seen)
        for s, s_prime in pairs:
            read_idx = (i + s) % N
            write_idx = (i * a_inv + s_prime) % N
            output[write_idx] = C[read_idx]
        for s, _ in pairs:
            target_idx = (i + s) % N
            if not seen[target_idx]:
                seen[target_idx] = True
                remaining -= 1

    intercept = random.randint(0, N - 1)
    output = output[N - intercept:] + output[:N - intercept]
    if __debug__:
        assert None not in output
        assert Counter(output) == Counter(C)
    return output
