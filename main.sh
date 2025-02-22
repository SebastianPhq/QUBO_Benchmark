#!/bin/bash

DATA_DIR="./data"                   # Data directory located in the current directory's "data" folder
MODEL_DIR="."                      # Directory for model files
ZIMPL_EXEC="./executable/zimpl-3.6.1.linux.x86_64.gnu.static.opt"  # ZIMPL executable located in the "executable" folder
QOQO_EXEC="./executable/qoqo-2.0.1-gcc-13-opt-omp-all-none-sm_86"    # qoqo executable located in the "executable" folder

# Create directories for storing results and intermediate files
mkdir -p log lp_files bqp_solutions eval_solutions qs_files qoqo_solutions mst_files

# Define an array of q values (stored as strings)
q_values=("0" "0.000001" "0.00001" "0.00005" "0.0001" "0.0005" "0.001" "0.01")

# Define the time interval corresponding to each asset count
declare -A t_values
t_values[200]=10
t_values[499]=15

# Loop over each asset count
for a in 200 499; do
  # Select the appropriate data files based on asset count
  if [ "$a" -eq 200 ]; then
    stock_price_data="data/2023prices_200.txt.gz"
    covariance_data="data/2023covmat_200.txt.gz"
  elif [ "$a" -eq 499 ]; then
    stock_price_data="data/2024prices_499.txt.gz"
    covariance_data="data/2024covmat_499.txt.gz"
  fi

  # Set the time interval t
  t=${t_values[$a]}

  echo "Processing asset value: $a with time intervals: $t"

  # Loop over each q value
  for q in "${q_values[@]}"; do
    echo "Running with a=$a, t=$t, q=$q"

    # Format asset count and time interval with leading zeros
    a_padded=$(printf "%03d" "$a")
    t_padded=$(printf "%02d" "$t")

    ###############################
    # 1. BQP Model Generation and Solving Section
    ###############################
    base_filename="bqp_a${a_padded}_t${t_padded}_q${q}"
    lp_file="lp_files/${base_filename}"
    eval_lp_file="lp_files/bqp_eval_a${a_padded}_t${t_padded}_q${q}"
    solution_file="bqp_solutions/${base_filename}.sol"
    eval_solution_file="eval_solutions/bqp_eval_a${a_padded}_t${t_padded}_q${q}.sol"

    # Use ZIMPL to generate the LP file for the BQP model (a .tbl file with the same name is also generated)
    $ZIMPL_EXEC \
      -Dtime_intervals="$t" \
      -Dq="$q" \
      -Dstock_price="$stock_price_data" \
      -Dstock_covariance="$covariance_data" \
      -o "$lp_file" \
      "$MODEL_DIR/bqp_u3_c10.zpl"

    $ZIMPL_EXEC \
      -Dtime_intervals="$t" \
      -Dq="$q" \
      -Dstock_price="$stock_price_data" \
      -Dstock_covariance="$covariance_data" \
      -o "$eval_lp_file" \
      "$MODEL_DIR/bqp_eval_u3_c10.zpl"

    # Process and compress the generated LP files
    sed -i 's/\.[^ ]*//g' "$lp_file.lp"
    sed -i 's/\.[^ ]*//g' "$eval_lp_file.lp"
    gzip -f "$lp_file.lp"
    gzip -f "$eval_lp_file.lp"

    # Solve the BQP model using Gurobi
    gurobi_cl LogFile="log/${base_filename}.log" TimeLimit=10000 MIPGap=0 ResultFile="$solution_file" "$lp_file.lp.gz"

    # Evaluate the BQP solution using Gurobi
    gurobi_cl LogFile="log/eval_${base_filename}.log" TimeLimit=100 MIPGap=0 ResultFile="$eval_solution_file" InputFile="$solution_file" "$eval_lp_file.lp.gz"

    #########################################
    # 2. QS Model Generation and qoqo Solving Section
    #########################################
    # Define the base name for the QS model
    qs_base="uqo_a${a_padded}_t${t_padded}_q${q}"
    qs_file="qs_files/${qs_base}"

    # Use ZIMPL to generate the QS model file (using the -t q option), keeping the same parameters as BQP
    $ZIMPL_EXEC \
      -Dtime_intervals="$t" \
      -Dq="$q" \
      -Dstock_price="$stock_price_data" \
      -Dstock_covariance="$covariance_data" \
      -o "$qs_file" \
      -t q \
      "$MODEL_DIR/uqo_u3_c10.zpl"

    # Process and compress the QS file
    sed -i 's/\.[^ ]*//g' "$qs_file.qs"
    gzip -f "$qs_file.qs"

    # Run qoqo to solve the QS model (fixed time limit of 3600 seconds)
    mst_file="mst_files/${qs_base}"
    qoqo_log="log/${qs_base}.log"
    $QOQO_EXEC -O8 -T3600 -f1 -o "$mst_file" -v 2 "$qs_file.qs.gz" >> "$qoqo_log" 2>&1

    # Do not directly copy qoqo's solution; a Python script will convert the .mst file to a .sol file
  done
done

# Call the Python script to convert the qoqo-generated .mst files into .sol files for evaluation
python3 convert_mst_to_sol.py

#########################################
# 3. Evaluation of the Converted qoqo Solutions
#########################################
# For each asset count and q value, evaluate the converted qoqo solution using the previously generated evaluation LP model
for a in 200 499; do
  # Set the corresponding time interval
  if [ "$a" -eq 200 ]; then
    t=10
  else
    t=15
  fi
  a_padded=$(printf "%03d" "$a")
  t_padded=$(printf "%02d" "$t")
  for q in "${q_values[@]}"; do
    qs_base="uqo_a${a_padded}_t${t_padded}_q${q}"
    qoqo_sol="qoqo_solutions/${qs_base}.sol"
    eval_lp_file="lp_files/bqp_eval_a${a_padded}_t${t_padded}_q${q}.lp.gz"
    out_file="eval_solutions/uqo_eval_a${a_padded}_t${t_padded}_q${q}.sol"

    echo "Evaluating qoqo solution: $qoqo_sol using LP model $eval_lp_file"
    gurobi_cl LogFile="log/eval_${qs_base}.log" TimeLimit=100 MIPGap=0 \
      ResultFile="$out_file" \
      InputFile="$qoqo_sol" "$eval_lp_file"
  done
done