# bqp_eval_u3_c10.zpl
# Evaluation LP and QUBO model objective, risk, profit and trans_fee, using the solution obtained from the main model

include "parameter_u3_c10.zpl";

var risk >= -infinity <= infinity;
var profit >= -infinity <= infinity;
var trans_fee >= -infinity <= infinity;

minimize obj: q * risk - profit;

subto risk:
   risk == upscale * (
      sum <t> in TX : (
         sum <i,m,sl1,j,n,sl2> in SX*SX :
            (sl1 * sl2 * cov[i,j,t] * x[i,m,sl1,t] * up[i,t]
                              * x[j,n,sl2,t] * up[j,t])
      )
   );

subto profit:
   profit == upscale * (
      sum <t> in TX : (
           rho_c * unit * sum <k> in CS1 : 2^k * s1[k,t] # cash interest
         - rho_s * sum <i,m,-1> in SX : up[i,t] * x[i,m,-1,t] # short sell interest
      )
      + sum <t> in TX \ { t_beg } : (
          sum <i,m,sl> in SX : sl * (up[i,t+1] - up[i,t]) * x[i,m,sl,t]  #profit
      )
      + sum <i,m,sl> in SX : sl * (up[i,t_beg+1] - up[i,t_beg]) #profit for the first day
                                * x[i,m,sl,t_beg]
      - trans_fee
   );

subto trans_fee:
   trans_fee ==
      sum <t> in TX \ { t_beg } : (
         sum <i,m,sl> in SX :
            delta * up[i,t] * (x[i,m,sl,t-1] + x[i,m,sl,t]
                               - 2 * x[i,m,sl,t-1] * x[i,m,sl,t]) #transaction fee
      )
      + sum <i,m,sl> in SX :
            delta * up[i,t_beg] * x[i,m,sl,t_beg] #transaction fee for the first day
      + delta * sum <i,m,sl> in SX :
            up[i,t_end] * x[i,m,sl,t_end];  #transaction fee for the last day

subto c2:
   forall <t> in TX :
      sum <i,m,sl> in SX : sl * x[i,m,sl,t] + sum <k> in CS1 : 2^k * s1[k,t] == b_csh;

subto c3:    # Total number of assets (including short sells) must not exceed b_tot
   forall <t> in TX :
      sum <i,m,sl> in SX : x[i,m,sl,t] + sum <k> in CS2 : 2^k * s2[k,t] == b_tot;