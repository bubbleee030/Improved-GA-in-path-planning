# Improved genetic algorithm for path planning: a demonstration

**English** · [繁體中文](README.zh-TW.md)

Final project for a course on computational intelligence. It studies the improved genetic algorithm
of Di Chen (2024), which targets two weaknesses of a plain GA in path planning: slow convergence and
jagged paths. The paper's remedy is to change the operator rates over the run and to reward smooth
turns in the fitness function.

`demo.py` implements the paper's core mechanisms in isolation:

| Mechanism | Paper | Formula in `demo.py` |
|---|---|---|
| Non-linear crossover rate | Eq. 10 | `p_c(i) = p_d − √i / (1 + i)²` |
| Decreasing mutation rate | Eq. 11 | `p_m(i) = p_min · (1 − √(i / G))` |
| Smoothness penalty | Eq. 8–9 | Penalizes each turn by its angle. |

Running it plots both rate schedules over 50 generations and scores a sample path.

## Scope

This is an illustration of the mechanisms, not a full path planner. It has no population,
selection or map. The penalty weights in `calculate_smoothness_penalty` are simplified stand-ins
for the paper's α, β, γ values.

## Run

```bash
pip install numpy matplotlib      # or: pip install -r requirements.txt
python demo.py
```
