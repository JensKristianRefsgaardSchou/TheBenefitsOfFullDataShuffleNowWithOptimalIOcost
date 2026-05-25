import math
import random
from collections import Counter


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
