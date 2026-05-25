# The benefits of full data shuffle, now with optimal I/O cost

This repository contains the Python code and saved experiment output for the
paper's shuffle comparison experiment.

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

Implemented algorithms:

- `Gen-2-Wise-Ind-Perm` from Algorithm 1
- `IO Shuffle` from Algorithm 2, for one or two rounds
- `CorgiPile` baseline
- `Fisher-Yates` uniform shuffle baseline
