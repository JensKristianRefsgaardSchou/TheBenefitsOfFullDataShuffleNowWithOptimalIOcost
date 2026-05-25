import csv
import math
import multiprocessing as mp
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from time import time

from shuffle_functions import corgi_pile
from shuffle_functions import fisher_yates
from shuffle_functions import gen_2_wise_ind_perm
from shuffle_functions import io_shuffle


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
