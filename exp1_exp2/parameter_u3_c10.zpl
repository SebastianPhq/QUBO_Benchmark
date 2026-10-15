#############################################################################
# parameter_u3_c10.zpl
# Only scenarios for a in {200, 499} are considered.
#############################################################################

param time_intervals    := 10;  # Default t (overwritten by -Dt=... if desired)

# Basic parameters
param cash    := 1000000;
param unit    := 100000;
param delta   := 0.001;
param rho_p   := 0.0003;
param rho_c   := 0.0001;
param rho_s   := 0.000025;
param ub      := 3;
param q       := 1;
param upscale := 1;
param b_tot := 60;

# Start and end time indices
param t_beg := 6;
param t_end := t_beg + time_intervals - 1;

# How many units of stocks can be bought initially
param b_csh := cash / unit;

# Reading the data sets (stock prices and covariance)
set T               := { read stock_price as "<1n>" };
set S               := { read stock_price as "<2s>" };
param p[S*T]        := read stock_price as "<2s,1n> 3n";
param cov[S*S*T]    := read stock_covariance as "<2s,3s,1n> 4n";

# One unit is unit/p[s,t_beg] shares of stock s
param ucnt[<s> in S] := unit / p[s,t_beg];
# up[s,tt] is the price of one unit of stock s at time tt
param up[<s,tt> in S*T] := p[s,tt] * ucnt[s];

# Additional sets for slack variables
set CS1 := { 0 .. 3 };
set CS2 := { 0 .. 4 };
set TX  := { t_beg .. t_end };
set SC  := { 1 .. ub };
set SL  := { 1, -1 };
set SX  := S * SC * SL;

# Declare binary variables
var x[SX*TX]   binary;
var s1[CS1*TX] binary;
var s2[CS2*TX] binary;

# Optional info
do print card(SX)*card(TX);
do print card(CS1)*card(TX);
do print card(CS2)*card(TX);
