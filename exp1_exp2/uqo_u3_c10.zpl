# Main QUBO optimization model

include "parameter_u3_c10.zpl";

minimize risk:
   upscale * (
      sum <t> in TX : (
         q * sum <i,m,sl1,j,n,sl2> in SX*SX:
               (sl1 * sl2 * cov[i,j,t] * x[i,m,sl1,t] * up[i,t]  #risk
                                 * x[j,n,sl2,t] * up[j,t])
         - rho_c * unit * sum <k> in CS1 : 2^k * s1[k,t]  #cash interest
         + rho_s * sum <i,m,-1> in SX : up[i,t] * x[i,m,-1,t]  #short sell interest
      )
      + sum <t> in TX \ { t_beg } : (
          - sum <i,m,sl> in SX : (
              sl * (up[i,t+1] - up[i,t]) * x[i,m,sl,t]  #profit
            - delta * up[i,t] * (x[i,m,sl,t-1] + x[i,m,sl,t]
                                - 2 * x[i,m,sl,t-1] * x[i,m,sl,t])  #transaction fee
          )
       )
       - sum <i,m,sl> in SX : (
           sl * (up[i,t_beg+1] - up[i,t_beg]) * x[i,m,sl,t_beg]  #profit for the first day
         - delta * up[i,t_beg] * x[i,m,sl,t_beg]  #transaction fee for the first day
       )
       + delta * sum <i,m,sl> in SX : up[i,t_end] * x[i,m,sl,t_end]  #transaction fee for the last day
   )
;

subto c2:
   # Limit for short selling: total short units plus slack must match b_csh
   forall <t> in TX :
      sum <i,m,sl> in SX : sl * x[i,m,sl,t] + sum <k> in CS1 : 2^k * s1[k,t] == b_csh, qubo, penalty7;

subto c3:
   # Total number of assets (including short sells) must not exceed b_tot
   forall <t> in TX :
      sum <i,m,sl> in SX : x[i,m,sl,t] + sum <k> in CS2 : 2^k * s2[k,t] == b_tot, qubo, penalty7;