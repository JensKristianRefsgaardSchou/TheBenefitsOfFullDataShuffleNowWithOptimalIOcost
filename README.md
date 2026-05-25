# The benefits of full data shuffle, now with optimal I/O cost

This repository contains the Python code and saved experiment output for the
paper's shuffle comparison experiment.

Repository layout:

```text
shufflePaperExp.py                 Python experiment runner and result printer
shuffle_functions/                 One Python file per shuffle algorithm
cpp/run_experiment.py              C++ accelerated experiment runner
cpp/shuffle_core.cpp               C++ implementations used by the runner
twoWiseExp_counts_B400.csv         Saved data summarized in Table 2
Paper.pdf                          Accepted paper
```

Run:

```powershell
python shufflePaperExp.py
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
python -c "import shufflePaperExp as s; s.twoWiseExp(n=10007, B=14, reps=10, processes=4, workers_per_func=1, csv_path='new_counts.csv', reset_csv=True)"
```

To summarize the saved data with the C++ runner's reporting code:

```powershell
python cpp\run_experiment.py --print-only --csv twoWiseExp_counts_B400.csv --N 16000057 --B 400
```

To run the accelerated C++ implementation, first make sure a C++17 compiler is
available, then run:

```powershell
python cpp\run_experiment.py --N 10007 --B 14 --reps 10 --reset-csv
```

Implemented algorithms:

- `Gen-2-Wise-Ind-Perm` from Algorithm 1
- `IO Shuffle` from Algorithm 2, for one or two rounds
- `CorgiPile` baseline
- `Fisher-Yates` uniform shuffle baseline
