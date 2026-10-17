#!/usr/bin/env python3
"""Bias-field warm-start QAOA (Algorithm 1) on every Exp3 instance.

--mode simulator: noiseless exact simulation on the local machine (about 5 s per instance).
--mode ibm:       real hardware; needs IBM_API_KEY and IBM_INSTANCE_CRN in the environment.
                  Each run submits 41 jobs (about 126 s of QPU time).
Runs are written to runs/<mode>; rerunning the same command resumes an interrupted run
without resubmitting finished jobs. The runs of the paper are in results/.
"""
import argparse, json, os
from pathlib import Path
from portfolio import ROOT, instances
from portfolio.qaoa import run

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--mode', choices=['simulator', 'ibm'], required=True)
parser.add_argument('--backend', default='ibm_pittsburgh')
parser.add_argument('--seed', type=int, help='overrides the seed in qaoa_config.json')
parser.add_argument('--only', nargs='*', help='instance identifiers (default: all eight)')
parser.add_argument('--output', help='default: runs/<mode>')
args = parser.parse_args()
cfg = json.loads((ROOT/'qaoa_config.json').read_text())
if args.seed is not None: cfg['seed'] = args.seed
out = Path(args.output) if args.output else ROOT/'runs'/args.mode
token = instance = None
if args.mode == 'ibm':
    token, instance = os.environ['IBM_API_KEY'], os.environ['IBM_INSTANCE_CRN']
for name, problem in instances(args.only):
    result = run(problem, out/name, args.mode, cfg, args.backend if args.mode == 'ibm' else None, token, instance)
    print(f"{name}: best feasible objective {result['best_feasible_objective']:.2f}, "
          f"first found in {result['best_first_found_in']}, {result['shots']} shots")
