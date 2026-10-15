# QUBO Benchmark for Multi-Period Portfolio Optimization

Benchmark instances and code for multi-period portfolio optimization with short selling, transaction costs, cash interest, and integer position blocks. Each instance is formulated as a QUBO and as an equivalent constrained binary quadratic program (BQP). The relaxation-based lower bounds of the BQP provide a common reference for the solutions of classical, quantum-inspired, and quantum solvers.

| Folder | Instances | Solvers | Results in the paper |
|---|---|---|---|
| [`exp1_exp2/`](exp1_exp2) | Exp1: 200 S&P 500 stocks, 10 holding intervals (12,100 binary variables)<br>Exp2: 499 stocks, 15 holding intervals (45,060 binary variables) | Gurobi on the BQP, qoqo/ABS2 on the QUBO | Tables 3 and 4 |
| [`exp3/`](exp3) | Exp3: 10 stocks, 6 holding intervals (156 binary variables) | Dynamic programming, Gurobi, warm-start QAOA on IBM quantum hardware, noiseless simulator | Table 6 |

All experiments use the eight risk-aversion levels q = 0, 10⁻⁶, 10⁻⁵, 5·10⁻⁵, 10⁻⁴, 5·10⁻⁴, 10⁻³, and 10⁻². Each folder has its own README with requirements and instructions.

## License

MIT, see [LICENSE](LICENSE).
