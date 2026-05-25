#!/usr/bin/env python3
"""
Two-wise pair experiment — C++ accelerated.
Works on Windows (MSVC or MinGW) and Linux/macOS (g++/clang++).

Usage:
    python run_experiment.py                    # run with defaults
    python run_experiment.py --N 100003 --B 140 --reps 250
    python run_experiment.py --N 16000000 --B 4000 --reps 50
    python run_experiment.py --print-only --csv results.csv   # just print saved results
"""

import argparse
import ctypes
import csv
import os
import platform
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import time

# ── Platform detection ───────────────────────────────────────────────────────

IS_WINDOWS = platform.system() == "Windows"
LIB_EXT = ".dll" if IS_WINDOWS else ".so"
CPP_PATH = Path(__file__).parent / "shuffle_core.cpp"
LIB_PATH = Path(__file__).parent / f"shuffle_core{LIB_EXT}"

FUNC_NAMES = [
    "Gen-2-Wise-Ind-Perm",
    "IO Shuffle (1 round)",
    "IO Shuffle (2 rounds)",
    "CorgiPile (1 round)",
    "Fisher-Yates",
]

# ── Build ────────────────────────────────────────────────────────────────────

def _find_compiler():
    """Return (compiler_id, cmd_list) or raise RuntimeError."""
    # Try g++ first (Linux, macOS, MinGW on Windows)
    gpp = shutil.which("g++")
    if gpp:
        return "g++", [
            gpp, "-O3", "-shared", "-fPIC",
            "-std=c++17", "-pthread", "-static",
            str(CPP_PATH), "-o", str(LIB_PATH),
        ]

    # Try clang++
    clang = shutil.which("clang++")
    if clang:
        return "clang++", [
            clang, "-O3", "-shared", "-fPIC",
            "-std=c++17", "-pthread",
            str(CPP_PATH), "-o", str(LIB_PATH),
        ]

    if IS_WINDOWS:
        # Try MSVC cl.exe
        cl = shutil.which("cl")
        if cl:
            return "cl", [
                cl, "/O2", "/EHsc", "/std:c++17", "/LD",
                str(CPP_PATH), f"/Fe:{LIB_PATH}",
            ]

    raise RuntimeError(
        "No C++ compiler found.\n"
        "Install one of:\n"
        "  - MinGW-w64:  winget install -e --id MSYS2.MSYS2\n"
        "    then in MSYS2:  pacman -S mingw-w64-x86_64-gcc\n"
        "    and add C:\\msys64\\mingw64\\bin to PATH\n"
        "  - Visual Studio Build Tools:\n"
        "    https://visualstudio.microsoft.com/visual-cpp-build-tools/\n"
        "    then run from 'Developer Command Prompt'"
    )


def build(force=False):
    if not force and LIB_PATH.exists() and LIB_PATH.stat().st_mtime > CPP_PATH.stat().st_mtime:
        return
    compiler_id, cmd = _find_compiler()
    print(f"Compiling shuffle_core.cpp with {compiler_id} …")
    subprocess.check_call(cmd)
    print("Done.")


def load_lib():
    lib = ctypes.CDLL(str(LIB_PATH))
    lib.run_worker.argtypes = [
        ctypes.c_int,                            # func_id
        ctypes.c_int,                            # N
        ctypes.c_int,                            # B
        ctypes.c_int,                            # reps
        ctypes.POINTER(ctypes.c_longlong),       # out_counts
    ]
    lib.run_worker.restype = None

    lib.run_single_shuffle.argtypes = [
        ctypes.c_int,                            # func_id
        ctypes.c_int,                            # N
        ctypes.c_int,                            # B
        ctypes.POINTER(ctypes.c_int),            # out
    ]
    lib.run_single_shuffle.restype = None

    lib.get_num_funcs.argtypes = []
    lib.get_num_funcs.restype = ctypes.c_int
    return lib


# ── CSV helpers ──────────────────────────────────────────────────────────────

def append_csv(path, func_name, counts):
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerows((func_name, c) for c in counts)
        f.flush()


def print_results(path, N=None, B=None):
    counts_by_name = {n: [] for n in FUNC_NAMES}
    if not os.path.exists(path):
        print(f"No results file: {path}")
        return
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            name = row[0]
            if name not in counts_by_name:
                continue
            try:
                counts_by_name[name].append(int(row[1]))
            except ValueError:
                continue

    total = sum(len(v) for v in counts_by_name.values())
    parts = []
    if N is not None: parts.append(f"N={N}")
    if B is not None: parts.append(f"B={B}")
    parts.append(f"total_reps={total}")
    print(", ".join(parts))

    for name in FUNC_NAMES:
        c = sorted(counts_by_name[name])
        if not c:
            print(f"  {name}: no samples")
            continue
        n = len(c)
        mean = sum(c) / n
        median = c[n // 2] if n % 2 else (c[n // 2 - 1] + c[n // 2]) / 2
        q25 = c[int(0.25 * (n - 1))]
        q75 = c[int(0.75 * (n - 1))]
        print(f"  {name}: reps={n}  mean={mean:.2f}  median={median}  q25={q25}  q75={q75}")


# ── Worker (runs in a child process) ─────────────────────────────────────────

def _worker(func_id: int, N: int, B: int, reps: int,
            batch_size: int, csv_path: str):
    """Each process loads the .dll/.so independently -> separate RNG state.
    Writes results to CSV every batch_size reps so progress survives crashes."""
    lib = load_lib()
    func_name = FUNC_NAMES[func_id]
    done = 0
    while done < reps:
        chunk = min(batch_size, reps - done)
        buf = (ctypes.c_longlong * chunk)()
        lib.run_worker(func_id, N, B, chunk, buf)
        counts = list(buf)
        append_csv(csv_path, func_name, counts)
        done += chunk
    return func_id, done


# ── Main ─────────────────────────────────────────────────────────────────────

def run_experiment(N, B, reps, processes, workers_per_func, csv_path, reset_csv,
                   batch_size=10):
    build()

    num_funcs = len(FUNC_NAMES)
    max_workers = min(processes, num_funcs * workers_per_func)

    tasks = []
    for fid in range(num_funcs):
        base = reps // workers_per_func
        rem  = reps % workers_per_func
        for w in range(workers_per_func):
            r = base + (1 if w < rem else 0)
            if r > 0:
                tasks.append((fid, r))

    if reset_csv:
        open(csv_path, "w").close()

    print(f"Running: N={N}, B={B}, reps={reps}, processes={max_workers}, "
          f"tasks={len(tasks)}, batch_size={batch_size}")
    t0 = time()

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_worker, fid, N, B, r, batch_size, csv_path): (fid, r)
            for fid, r in tasks
        }
        done = 0
        try:
            for fut in as_completed(futures):
                fid, reps_done = fut.result()
                done += 1
                if done % max(1, len(tasks) // 10) == 0:
                    elapsed = time() - t0
                    print(f"  {done}/{len(tasks)} tasks done  ({elapsed:.1f}s)")
        except KeyboardInterrupt:
            print("\nInterrupted — partial results:")
            pool.shutdown(cancel_futures=True, wait=False)

    elapsed = time() - t0
    print(f"\nFinished in {elapsed:.1f}s")
    print_results(csv_path, N=N, B=B)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=100003)
    p.add_argument("--B", type=int, default=140)
    p.add_argument("--reps", type=int, default=250)
    p.add_argument("--processes", type=int, default=os.cpu_count() or 4)
    p.add_argument("--workers-per-func", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=10,
                   help="Write results to CSV every N reps per worker (crash safety)")
    p.add_argument("--csv", type=str, default="twoWiseExp_counts_cpp.csv")
    p.add_argument("--reset-csv", action="store_true")
    p.add_argument("--print-only", action="store_true")
    p.add_argument("--build-only", action="store_true")
    args = p.parse_args()

    if args.build_only:
        build(force=True)
        return

    if args.print_only:
        print_results(args.csv, N=args.N, B=args.B)
        return

    run_experiment(
        N=args.N, B=args.B, reps=args.reps,
        processes=args.processes,
        workers_per_func=args.workers_per_func,
        csv_path=args.csv,
        reset_csv=args.reset_csv,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
