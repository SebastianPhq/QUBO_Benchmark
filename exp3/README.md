# Exp3: gate-based quantum hardware, Gurobi, and noiseless simulator

Code and data to reproduce Table 6 of the paper: the instance `n10_T6_k1_B3_C2` (10 stocks, 6 holding intervals, 156 binary variables, 120 qubits) for eight risk-aversion levels q, solved by

- dynamic programming (certified optimum),
- Gurobi on the BQP (LB, UB, Gap, Gurobi TTS),
- warm-start QAOA on IBM Heron r3 `ibm_pittsburgh` (IBM TTS, optimal shots),
- the same algorithm on a noiseless simulator (Simulator TTS, optimal shots).

## Quick start

```bash
pip install -r requirements.txt
python make_table.py          # Table 6 from the results in results/
```

## Contents

| Path | Content |
|---|---|
| `instances/` | The eight instances (prices, covariances, QUBO and BQP matrices). Stocks AXON, ORCL, TDY, MAA, VST, IBM, DFS, ON, FTV, NVR from the Exp2 data in `../exp1_exp2/data/`, starting on its first date. |
| `portfolio/model.py` | Objective and constraints of the instances |
| `portfolio/exact.py` | Dynamic programming and Gurobi |
| `portfolio/qaoa.py` | Bias-field warm-start QAOA (Algorithm 1): simulator and IBM hardware |
| `qaoa_config.json` | Hyperparameters: one layer, γ = 2, β = −0.6, 4 replicas, 40 rounds of 1,000 shots and a final job of 4,000 shots per replica (176,000 shots per run) |
| `run_dp.py`, `run_gurobi.py`, `run_qaoa.py` | Experiments; each writes to `results/` |
| `make_table.py` | Table 6 from `results/` |
| `results/` | Results reported in the paper, including the raw counts, job IDs, job metrics, compiled circuits, and calibration snapshots of the hardware runs |

## Reproducing the columns

To rerun an experiment, delete its folder in `results/` first (the QAOA script resumes existing runs), then run `python make_table.py`.

| Columns | Command | Run time |
|---|---|---|
| UB check (certified optimum) | `python run_dp.py` | < 1 s |
| LB, UB, Gap, Gurobi TTS | `python run_gurobi.py` | 6 instances in seconds; q = 0.0005 and q = 0.001 run to the 10,000 s limit and need several GB of memory each |
| Simulator TTS and optimal shots | `python run_qaoa.py --mode simulator` | about 5 s per instance |
| IBM TTS and optimal shots | `python run_qaoa.py --mode ibm` | 41 jobs and about 126 s of QPU time per instance |

- Gurobi 13.0.3 with one thread, seed 101, relative gap tolerance 0, and the all-cash allocation as the only start. The instances are small enough for the size-limited license that comes with `pip install gurobipy`.
- The simulator samples are deterministic: a rerun returns exactly the same counts. Times depend on the machine.
- The hardware mode needs an IBM Quantum account: set `IBM_API_KEY` and `IBM_INSTANCE_CRN`. Hardware samples differ from run to run because of noise and calibration; `results/ibm/` contains the runs of the paper (2 October 2026, Qiskit Runtime 0.50, no error mitigation). The seed of each run is recorded in its `manifest.json`.

## Definitions

- **Gurobi TTS**: runtime at which the final incumbent was first found.
- **IBM TTS**: QPU time (`qpu_charge_time_seconds` of each job) plus the local bias updates between jobs, up to the job in which the optimum was first sampled. It excludes queueing, data transfer, cloud-side processing, and the one-time setup.
- **Simulator TTS**: time from the start of the run to the end of the round in which the optimum was first sampled.
- **Optimal shots**: shots that returned the optimum, over all 176,000 shots of a run.
- **Gap**: \|UB − LB\| / \|UB\|. The UB of every row equals the optimum certified by dynamic programming.
