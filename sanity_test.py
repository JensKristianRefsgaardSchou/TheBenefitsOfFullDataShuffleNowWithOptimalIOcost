import ctypes
import importlib.util
import os
import random
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON_DIR = ROOT / "python"
CPP_DIR = ROOT / "cpp"
N = 101
B = 5
SEED = 12345


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assert_permutation(output, expected):
    assert len(output) == len(expected)
    assert Counter(output) == Counter(expected)


def pairs_in_same_output_block(output, B):
    count = 0
    for i in range((len(output) + B - 1) // B):
        block_origins = [x // B for x in output[i * B:(i + 1) * B]]
        freq = Counter(block_origins)
        count += sum(v * (v - 1) // 2 for v in freq.values())
    return count


def test_python_implementation():
    sys.path.insert(0, str(PYTHON_DIR))
    shuffle_core = load_module("python_shuffle_core", PYTHON_DIR / "shuffle_core.py")
    python_runner = load_module("python_run_experiment", PYTHON_DIR / "run_experiment.py")
    data = list(range(N))
    functions = [
        ("Gen-2-Wise-Ind-Perm", lambda x: shuffle_core.gen_2_wise_ind_perm(x, B=B)),
        ("IO Shuffle (1 round)", lambda x: shuffle_core.io_shuffle(x, B=B, rounds=1)),
        ("IO Shuffle (2 rounds)", lambda x: shuffle_core.io_shuffle(x, B=B, rounds=2)),
        ("CorgiPile (1 round)", lambda x: shuffle_core.corgi_pile(x, B=B)),
        ("Fisher-Yates", lambda x: shuffle_core.fisher_yates(x)),
    ]

    assert python_runner.PAPER_ALGORITHM_NAMES == [name for name, _ in functions]
    for name, func in functions:
        random.seed(SEED)
        first = func(data)
        random.seed(SEED)
        second = func(data)
        assert first == second, f"{name} is not deterministic under random.seed"
        assert_permutation(first, data)
        count = pairs_in_same_output_block(first, B)
        assert 0 <= count <= len(data) * (B - 1) // 2


def test_cpp_implementation():
    cpp_runner = load_module("cpp_run_experiment", CPP_DIR / "run_experiment.py")
    python_runner = load_module("python_run_experiment_for_names", PYTHON_DIR / "run_experiment.py")
    assert cpp_runner.FUNC_NAMES == python_runner.PAPER_ALGORITHM_NAMES

    cpp_runner.build()
    lib = cpp_runner.load_lib()
    assert lib.get_num_funcs() == len(cpp_runner.FUNC_NAMES)

    expected = list(range(N))
    for func_id, name in enumerate(cpp_runner.FUNC_NAMES):
        out = (ctypes.c_int * N)()
        lib.run_single_shuffle(func_id, N, B, out)
        output = list(out)
        assert_permutation(output, expected)
        count = pairs_in_same_output_block(output, B)
        assert 0 <= count <= len(output) * (B - 1) // 2, name


def main():
    test_python_implementation()
    test_cpp_implementation()
    print("sanity_test.py: all checks passed")
    print(
        "Note: exact Python-vs-C++ seeded output equality is not checked. "
        "The C++ core currently seeds mt19937_64 from random_device, while "
        "the Python core uses Python's random module."
    )


if __name__ == "__main__":
    main()
