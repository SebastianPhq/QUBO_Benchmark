"""Exact references: dynamic programming (certified optimum) and Gurobi on the BQP."""
from itertools import combinations
from time import perf_counter
import numpy as np


def solve_dp(problem, chunk=512):
    """Global optimum by dynamic programming over all feasible holdings per interval.

    Exact because the objective couples only consecutive intervals (rebalancing
    cost) and all constraints are per interval.
    """
    start = perf_counter()
    s = problem.spec
    nx = 2 * s.n * s.k
    sign = np.tile([1, -1], s.n * s.k)
    states = []
    for gross in range(min(s.B, nx) + 1):
        for ids in combinations(range(nx), gross):
            z = np.zeros(nx, dtype=np.int8); z[list(ids)] = 1
            if s.C - s.cash_cap <= z @ sign <= s.C:
                states.append(z)
    states = np.array(states)
    count = len(states)
    blocks = states.reshape(count, s.n, s.k, 2)
    net = (blocks[..., 0] - blocks[..., 1]).sum(axis=2)
    shorts = blocks[..., 1].sum(axis=2)
    cash = s.C - net.sum(axis=1)
    sf = states.astype(float)
    local = []
    for t in range(s.T):
        exposure = net * problem.up[t]
        risk = np.einsum('bi,ij,bj->b', exposure, problem.covariance[t], exposure)
        gain = net @ (problem.up[t+1] - problem.up[t])
        value = s.q * risk - gain + s.rho_s * (shorts @ problem.up[t]) - s.rho_c * s.unit * cash
        if t == 0:
            value += s.delta * (sf @ np.repeat(problem.up[t], 2 * s.k))      # opening trades
        if t == s.T - 1:
            value += s.delta * (sf @ np.repeat(problem.up[t+1], 2 * s.k))    # liquidation
        local.append(value)
    dp = local[0].copy()
    parents = []
    for t in range(1, s.T):
        fee = np.repeat(s.delta * problem.up[t], 2 * s.k)
        single, weighted = sf @ fee, sf * fee
        parent = np.empty(count, dtype=np.int64); best = np.empty(count)
        for j in range(0, count, chunk):
            block = slice(j, min(j + chunk, count))
            cand = dp[:, None] + single[:, None] + single[None, block] - 2 * weighted @ sf[block].T
            parent[block] = np.argmin(cand, axis=0)
            best[block] = cand[parent[block], np.arange(cand.shape[1])]
        parents.append(parent)
        dp = best + local[t]
    path = [int(np.argmin(dp))]
    for parent in parents[::-1]:
        path.append(int(parent[path[-1]]))
    x = problem.encode_positions(blocks[path[::-1]])
    objective = float(problem.energies(x, economic=True))
    if not np.isclose(objective, float(dp.min()), rtol=1e-10, atol=1e-5):
        raise RuntimeError('DP and BQP objective disagree')
    return {'objective': objective, 'states_per_interval': count,
            'solve_seconds': perf_counter() - start}


def solve_gurobi(problem, time_limit, seed=101, threads=1, log_file=None):
    """Gurobi on the BQP (economic objective subject to A x = b), all-cash start only.

    TTS is the runtime at which the final incumbent was first found (MIPSOL callback).
    """
    import gurobipy as gp
    from gurobipy import GRB
    incumbents = []

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            incumbents.append((model.cbGet(GRB.Callback.RUNTIME), model.cbGet(GRB.Callback.MIPSOL_OBJ)))

    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()
        with gp.Model('portfolio', env=env) as model:
            if log_file:
                model.Params.LogToConsole = 0
                model.Params.LogFile = str(log_file)
                model.Params.OutputFlag = 1
            model.Params.TimeLimit = time_limit
            model.Params.MIPGap = 0
            model.Params.MIPGapAbs = 1e-6
            model.Params.NonConvex = 2
            model.Params.Seed = seed
            model.Params.Threads = threads
            Q = problem.economic_Q
            x = model.addVars(problem.d, vtype=GRB.BINARY, name='x')
            model.setObjective(gp.quicksum(float(Q[i, j]) * x[i] * x[j] for i, j in zip(*np.nonzero(Q))),
                               GRB.MINIMIZE)
            for j, (a, b) in enumerate(zip(problem.A, problem.b)):
                model.addConstr(gp.quicksum(float(a[i]) * x[i] for i in np.flatnonzero(a)) == float(b),
                                name=f'budget_{j}')
            for i, value in enumerate(problem.all_cash()):
                x[i].Start = int(value)
            model.optimize(callback)
            bits = np.rint([x[i].X for i in range(problem.d)]).astype(np.int8)
            objective = float(model.ObjVal)
            if not problem.feasible(bits) or not np.isclose(objective, problem.energies(bits, economic=True),
                                                            rtol=1e-8, atol=1e-4):
                raise RuntimeError('Gurobi incumbent is infeasible or its objective disagrees')
            tol = 1e-6 * max(1.0, abs(objective))
            return {'status': 'OPTIMAL' if model.Status == GRB.OPTIMAL else
                              'TIME_LIMIT' if model.Status == GRB.TIME_LIMIT else str(model.Status),
                    'objective': objective, 'lower_bound': float(model.ObjBound),
                    'gap': abs(objective - model.ObjBound) / abs(objective),
                    'tts_seconds': next(t for t, obj in incumbents if abs(obj - objective) <= tol),
                    'runtime_seconds': float(model.Runtime), 'time_limit_seconds': time_limit,
                    'seed': seed, 'threads': threads, 'gurobi_version': '.'.join(map(str, gp.gurobi.version()))}
