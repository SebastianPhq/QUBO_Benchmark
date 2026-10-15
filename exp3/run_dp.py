#!/usr/bin/env python3
"""Certified optimum of every Exp3 instance by dynamic programming (milliseconds each)."""
import json
from portfolio import ROOT, instances
from portfolio.exact import solve_dp

out = ROOT/'results'/'dp'; out.mkdir(parents=True, exist_ok=True)
for name, problem in instances():
    result = solve_dp(problem)
    (out/f'{name}.json').write_text(json.dumps(result, indent=2))
    print(f"{name}: optimum {result['objective']:.2f} ({result['states_per_interval']} feasible holdings "
          f"per interval, {result['solve_seconds']:.3f} s)")
