import csv
import math
import multiprocessing as mp
import os
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from time import time


PAPER_ALGORITHM_NAMES = [
    "Gen-2-Wise-Ind-Perm",
    "IO Shuffle (1 round)",
    "IO Shuffle (2 rounds)",
    "CorgiPile (1 round)",
    "Fisher-Yates",
]

CSV_NAME_ALIASES = {
    "PeyShuffle": "Gen-2-Wise-Ind-Perm",
    "IO shuffle (1 round)": "IO Shuffle (1 round)",
    "IO shuffle (2 rounds)": "IO Shuffle (2 rounds)",
}


def uniform_shuffle(data):
    return random.sample(data, len(data))


def fisher_yates(data):
    result = list(data)
    for i in range(len(result) - 1, 0, -1):
        j = random.randint(0, i)
        result[i], result[j] = result[j], result[i]
    return result


def flatten(rows):
    result = []
    for row in rows:
        result.extend(row)
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


def pairs_in_same_output_block(output, B):
    count = 0
    for i in range(math.ceil(len(output) / B)):
        block_origins = [x // B for x in output[i * B:(i + 1) * B]]
        freq = Counter(block_origins)
        count += sum(v * (v - 1) // 2 for v in freq.values())
    return count


def _twowise_gen2wise(data, B):
    return gen_2_wise_ind_perm(data, B=B)


def _twowise_ioshuffle1(data, B):
    return io_shuffle(data, B=B, rounds=1)


def _twowise_ioshuffle2(data, B):
    return io_shuffle(data, B=B, rounds=2)


def _twowise_corgipile1(data, B):
    return corgi_pile(data, B=B)


def _twowise_fisher_yates(data, B):
    return fisher_yates(data)


_TWOWISE_FUNCS = [
    _twowise_gen2wise,
    _twowise_ioshuffle1,
    _twowise_ioshuffle2,
    _twowise_corgipile1,
    _twowise_fisher_yates,
]

_TWOWISE_NAMES = PAPER_ALGORITHM_NAMES


def _canonical_name(name):
    return CSV_NAME_ALIASES.get(name, name)


def _append_twowise_csv(csv_path, func_name, counts, lock):
    if not counts:
        return
    with lock:
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerows((func_name, c) for c in counts)


def print_twoWiseExp_results(csv_path, reorganized_path=None, print_summary=True, n=None, B=None):
    quantiles = (0.25, 0.75)
    counts_by_name = {name: [] for name in _TWOWISE_NAMES}
    if not os.path.exists(csv_path):
        print(f"No results file: {csv_path}")
        return
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            name = _canonical_name(row[0])
            if name not in counts_by_name:
                continue
            try:
                count = int(row[1])
            except ValueError:
                continue
            counts_by_name[name].append(count)
    if reorganized_path:
        with open(reorganized_path, "w", newline="") as f:
            writer = csv.writer(f)
            for name in _TWOWISE_NAMES:
                writer.writerow([name, *counts_by_name[name]])
    if not print_summary:
        return
    total_reps = sum(len(v) for v in counts_by_name.values())
    if n is not None or B is not None or total_reps:
        parts = []
        if n is not None:
            parts.append(f"{n = }")
        if B is not None:
            parts.append(f"{B = }")
        parts.append(f"total reps = {total_reps}")
        print(", ".join(parts))
    for name in _TWOWISE_NAMES:
        counts = counts_by_name[name]
        if counts:
            n_counts = len(counts)
            total = sum(counts)
            counts.sort()
            mid = n_counts // 2
            if n_counts % 2:
                median = counts[mid]
            else:
                median = (counts[mid - 1] + counts[mid]) / 2
            q_vals = [counts[int(q * (n_counts - 1))] for q in quantiles]
            print(
                name,
                f"reps={n_counts}",
                f"mean={total / n_counts}",
                f"median={median}",
                f"q25={q_vals[0]}",
                f"q75={q_vals[1]}",
            )
        else:
            print(name, "no samples")


def _twowise_worker(func_id, n, B, reps, write_every, csv_path, lock):
    total_pairs = 0
    func = _TWOWISE_FUNCS[func_id]
    func_name = _TWOWISE_NAMES[func_id]
    batch = []
    for _ in range(reps):
        X = list(range(n))
        shuffled = func(X, B)
        pairs = pairs_in_same_output_block(shuffled, B)
        total_pairs += pairs
        batch.append(pairs)
        if len(batch) >= write_every:
            _append_twowise_csv(csv_path, func_name, batch, lock)
            batch = []
    if batch:
        _append_twowise_csv(csv_path, func_name, batch, lock)
    return func_id, total_pairs, reps


def twoWiseExp(
    n=10007,
    B=14,
    reps=25000,
    processes=24,
    workers_per_func=4,
    checkpoint_every=100,
    write_every=5,
    csv_path="twoWiseExp_counts.csv",
    reset_csv=False,
    reorganized_path=None,
):
    if reps is None:
        reps = n
    func_count = len(_TWOWISE_FUNCS)
    workers_per_func = max(workers_per_func, 1)
    write_every = max(write_every, 1)
    max_workers = min(processes, func_count * workers_per_func)
    tasks = []
    for func_id in range(func_count):
        base = reps // workers_per_func
        remainder = reps % workers_per_func
        for worker_id in range(workers_per_func):
            reps_i = base + (1 if worker_id < remainder else 0)
            if reps_i:
                tasks.append((func_id, reps_i))
    if reset_csv:
        open(csv_path, "w").close()

    manager = mp.Manager()
    lock = manager.Lock()
    start = time()
    total_tasks = len(tasks)
    executor = ProcessPoolExecutor(max_workers=max_workers)
    try:
        futures = [
            executor.submit(_twowise_worker, func_id, n, B, reps_i, write_every, csv_path, lock)
            for func_id, reps_i in tasks
        ]
        completed = 0
        for fut in as_completed(futures):
            fut.result()
            completed += 1
            if checkpoint_every and completed % checkpoint_every == 0:
                elapsed = time() - start
                print(f"completed {completed}/{total_tasks} in {elapsed:.1f}s")
                print_twoWiseExp_results(csv_path, reorganized_path=reorganized_path, n=n, B=B)
    finally:
        executor.shutdown(wait=True, cancel_futures=False)
        manager.shutdown()
    print(f"{n = }, {reps = }, {processes = }, {workers_per_func = }, {write_every = }")
    print_twoWiseExp_results(csv_path, reorganized_path=reorganized_path, n=n, B=B)


if __name__ == "__main__":
    N = 16000057
    B = 400
    file = os.path.join(os.path.dirname(__file__), "twoWiseExp_counts_B400.csv")
    print_twoWiseExp_results(file, n=N, B=B)
