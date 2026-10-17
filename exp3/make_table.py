#!/usr/bin/env python3
"""Table 6 of the paper from results/{dp,gurobi,ibm,simulator}.

IBM TTS:       QPU time (qpu_charge_time_seconds) plus the local bias updates between
               jobs, up to the job in which the optimum was first sampled.
Simulator TTS: elapsed time from the start of the run to the end of that round.
Optimal shots: shots that returned the optimum, over all 176,000 shots of a run.
"""
import argparse, json
from datetime import datetime
from pathlib import Path
import numpy as np
from portfolio import ROOT, instances
from portfolio.qaoa import evaluate_counts, read_json

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--results', default=str(ROOT/'results'), help='folder with dp/ and gurobi/')
parser.add_argument('--ibm', help='hardware runs (default: results/ibm, the runs of the paper)')
parser.add_argument('--simulator', help='simulator runs (default: results/simulator)')
args = parser.parse_args()
res = Path(args.results)
ibm_dir = Path(args.ibm) if args.ibm else res/'ibm'
sim_dir = Path(args.simulator) if args.simulator else res/'simulator'
stamp = lambda s: datetime.fromisoformat(s.replace('Z', '+00:00'))


def scan(problem, run_dir, optimum):
    """Records in job order, label of the first record containing the optimum, optimal shots."""
    labels = [f'round_{r:03d}' for r in range(len(list((run_dir/'evaluations').glob('round_*'))))] + ['final']
    records = [read_json(run_dir/'evaluations'/f'{l}.json.gz') for l in labels]
    first, optimal, shots = None, 0, 0
    tol = 1e-6 * max(1.0, abs(optimum))
    for i, record in enumerate(records):
        for counts in record['counts']:
            _, _, w, economic, violation = evaluate_counts(problem, counts)
            hit = (violation == 0) & (np.abs(economic - optimum) <= tol)
            optimal += int(w[hit].sum()); shots += int(w.sum())
            if hit.any() and first is None: first = i
    return records, first, optimal, shots


def ibm_tts(records, first):
    qpu = sum(r['metrics']['usage']['qpu_charge_time_seconds'] for r in records[:first + 1])
    updates = sum((stamp(records[i + 1]['submitted_at_utc']) - stamp(records[i]['completed_at_utc'])).total_seconds()
                  for i in range(first))
    return qpu + updates


def money(v): return f"\\({v:,.2f}\\)".replace(',', '{,}').replace('\\(-', '\\(-')
def count(v): return f"{v:,}".replace(',', '{,}')
def seconds(v): return '\\(<\\)0.01' if v < 0.005 else f'{v:.2f}'


print(r'\(q\) & LB & UB & Gap & Gurobi & IBM & Simulator & IBM & Simulator \\')
for name, problem in instances():
    optimum = json.loads((res/'dp'/f'{name}.json').read_text())['objective']
    g = json.loads((res/'gurobi'/f'{name}.json').read_text())
    assert abs(g['objective'] - optimum) <= 1e-6 * max(1, abs(optimum)), f'{name}: Gurobi UB differs from DP optimum'
    hw_tts = hw_opt = sim_tts = sim_opt = '--'                   # runs that are missing stay empty
    if (ibm_dir/name/'result.json').exists():
        hw, first, opt, _ = scan(problem, ibm_dir/name, optimum)
        hw_tts, hw_opt = f'{ibm_tts(hw, first):.0f}', count(opt)
    if (sim_dir/name/'result.json').exists():
        sim, first, opt, _ = scan(problem, sim_dir/name, optimum)
        start = stamp(read_json(sim_dir/name/'manifest.json')['started_at_utc'])
        sim_tts, sim_opt = f"{(stamp(sim[first]['completed_at_utc']) - start).total_seconds():.2f}", count(opt)
    q = format(problem.spec.q, 'f').rstrip('0').rstrip('.') or '0'
    print(f"{q:8s} & {money(g['lower_bound'])} & {money(g['objective'])} & {100*g['gap']:.2f}\\% & "
          f"{seconds(g['tts_seconds'])} & {hw_tts} & {sim_tts} & {hw_opt} & {sim_opt} \\\\")
