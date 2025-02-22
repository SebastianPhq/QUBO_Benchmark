# BQP and QUBO Model Solver Workflow

This repository contains files for generating, solving, and evaluating Binary Quadratic Programming (BQP) and Quadratic Unconstrained Binary Optimization (QUBO) models using ZIMPL, Gurobi, and qoqo. The workflow includes generating LP files for the BQP model, solving the BQP model with Gurobi and evaluating its solution, generating QS models and solving them with qoqo, converting qoqo-generated MST files into SOL files with a Python script, and evaluating the converted qoqo solutions using the same LP evaluation model as the BQP model.

## Requirements
- GLIBC version 2.29 or higher.
- ABS2 Library: Download the appropriate ABS2 library from [https://abs2.cs.hiroshima-u.ac.jp/downloads](https://abs2.cs.hiroshima-u.ac.jp/downloads).
- Gurobi: Ensure that the Gurobi command-line solver (`gurobi_cl`) is installed and licensed.
- qoqo: Ensure that the qoqo executable is available.

## Installation
1. Install the ABS2 library in a suitable location (e.g., `~/QUBO_Benchmark`).
2. Set the library path based on your GPU model:
   - For V100:  
     `export LD_LIBRARY_PATH=~/QUBO_Benchmark/sm_70/abs2/lib:$LD_LIBRARY_PATH`
   - For A100:  
     `export LD_LIBRARY_PATH=~/QUBO_Benchmark/sm_80/abs2/lib:$LD_LIBRARY_PATH`
   - For GTX 4090:  
     `export LD_LIBRARY_PATH=~/QUBO_Benchmark/sm_86/abs2/lib:$LD_LIBRARY_PATH`
3. Download and install Gurobi and ensure that the `gurobi_cl` command-line solver is available.
4. Ensure that the executables for ZIMPL and qoqo are located in the `./executable/` folder or update the paths accordingly in the scripts.

## Files in the Repository
- **main.sh**: The main shell script that orchestrates the entire workflow. It generates LP files for the BQP model (and corresponding .tbl files), solves the BQP model using Gurobi, generates QS model files, runs qoqo to solve them, calls the Python script to convert qoqo-generated MST files into SOL files, and evaluates the converted qoqo solutions using the same evaluation LP model as used for the BQP model.
- **convert_mst_to_sol.py**: A Python script that converts the MST files generated by qoqo into SOL files suitable for evaluation. It uses the .tbl files generated by ZIMPL and the original BQP solution files as references.
- **bqp_eval_u3_c10.zpl**: Contains the evaluation LP model that defines the objective (based on risk, profit, and transaction fees) and the constraints for evaluating the solution obtained from the main BQP model.
- **bqp_u3_c10.zpl**: The primary LP model file that includes `parameter_u3_c10.zpl` and defines the optimization objective (incorporating risk, profit, and transaction fees) along with the necessary constraints for the BQP model.
- **parameter_u3_c10.zpl**: Defines the basic parameters for the models, including time intervals, cash, unit, delta, various coefficients (e.g., ρₚ, ρ₍c₎, ρ₍s₎), upper bound (`ub`), total buy-in (`b_tot`), and instructions for reading stock prices and covariance data.
- **uqo_u3_c10.zpl**: The QUBO model file that also includes `parameter_u3_c10.zpl` and defines the objective and constraints for the QUBO optimization, including penalty terms for constraint violations.

## Running the Program
To run the complete workflow, simply execute:
```bash
chmod +x main.sh
```
```bash
./main.sh
```