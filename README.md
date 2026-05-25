# The benefits of full data shuffle, now with optimal I/O cost

This repository contains the Python code and saved experiment output for the
paper's shuffle comparison experiment.

Repository layout:

```text
main.py                            Top-level runner for Python or C++ backends
python/run_experiment.py           Python experiment runner and result printer
python/shuffle_core.py             Python shuffle implementations
cpp/run_experiment.py              C++ accelerated experiment runner
cpp/shuffle_core.cpp               C++ implementations used by the runner
twoWiseExp_counts_B400.csv         Saved data summarized in Table 2
Paper.pdf                          Accepted paper
```

Run:

```powershell
python main.py
```

The default run summarizes `twoWiseExp_counts_B400.csv`, the saved data for the
Table 2 setting:

```text
N = 16000057
B = 400
target repetitions = 40000 per algorithm
```

To run a new, smaller experiment:

```powershell
python main.py python --N 10007 --B 14 --reps 10 --processes 4 --workers-per-func 1 --csv new_counts.csv --reset-csv
```

To summarize the saved data with the C++ runner's reporting code:

```powershell
python main.py cpp --print-only --csv twoWiseExp_counts_B400.csv --N 16000057 --B 400
```

To run the accelerated C++ implementation, first make sure a C++17 compiler is
available, then run:

```powershell
python main.py cpp --N 10007 --B 14 --reps 10 --reset-csv
```

Implemented algorithms:

- `Gen-2-Wise-Ind-Perm` from Algorithm 1
- `IO Shuffle` from Algorithm 2, for one or two rounds
- `CorgiPile` baseline
- `Fisher-Yates` uniform shuffle baseline
