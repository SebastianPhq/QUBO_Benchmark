#!/usr/bin/env python3
"""Gurobi on the BQP of every Exp3 instance: LB, UB, gap, and TTS (Table 6).

Paper setting: 10,000 s time limit, one thread, seed 101, relative gap tolerance 0.
Six instances finish within seconds; q = 0.0005 and q = 0.001 run to the time limit.
"""
import argparse, json
from pathlib import Path
from portfolio import ROOT, instances
from portfolio.exact import solve_gurobi

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--time-limit', type=float, default=10000)
parser.add_argument('--only', nargs='*', help='instance identifiers (default: all eight)')
parser.add_argument('--output', default=str(ROOT/'results'/'gurobi'))
args = parser.parse_args()
out = Path(args.output)
out.mkdir(parents=True, exist_ok=True)
for name, problem in instances(args.only):
    result = solve_gurobi(problem, args.time_limit, log_file=out/f'{name}.log')
    (out/f'{name}.json').write_text(json.dumps(result, indent=2))
    print(f"{name}: {result['status']}, UB {result['objective']:.2f}, LB {result['lower_bound']:.2f}, "
          f"gap {100*result['gap']:.2f}%, TTS {result['tts_seconds']:.4f} s")
