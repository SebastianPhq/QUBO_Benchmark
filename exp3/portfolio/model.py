"""Portfolio BQP/QUBO instance: loading, objective, and constraints.

T holding intervals [t, t+1], t = 1..T, with T+1 price dates. Variable order per
interval: position bits [asset, block, direction (+1, -1)], cash slack bits,
gross-position slack bits. E(x) = offset + x^T Q x with Q upper triangular.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import numpy as np


@dataclass(frozen=True)
class Spec:
    n: int
    T: int
    k: int = 1
    B: int = 3
    C: int = 2
    q: float = 0.0005
    penalty: float = 1e7
    unit: float = 100000.0
    delta: float = 0.001
    rho_c: float = 0.0001
    rho_s: float = 0.000025
    start: int = 6
    dataset: str = '2024'
    instance_seed: int = 11
    cash_bits: int = 4
    time_convention: str = 'paper_T_holding_intervals_T+1_price_dates_liquidate_at_p_T+1'

    def __post_init__(self):
        for name in ('q', 'penalty', 'unit', 'delta', 'rho_c', 'rho_s'):
            object.__setattr__(self, name, float(getattr(self, name)))

    @property
    def cash_cap(self):
        return 2**self.cash_bits - 1

    @property
    def gross_bits(self):
        return self.B.bit_length()

    @property
    def per_time(self):
        return 2 * self.n * self.k + self.cash_bits + self.gross_bits


@dataclass
class PortfolioQUBO:
    spec: Spec
    assets: list
    up: np.ndarray            # normalized prices, shape (T+1, n)
    covariance: np.ndarray    # shape (T, n, n)
    Q: np.ndarray             # penalized QUBO
    economic_Q: np.ndarray    # objective of the BQP
    A: np.ndarray             # equality constraints A x = b of the BQP
    b: np.ndarray
    offset: float

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as a:
            m = json.loads(str(a['metadata']))
            obj = cls(Spec(**m['spec']), m['assets'], a['up'], a['covariance'], a['Q'],
                      a['economic_Q'], a['A'], a['b'], m['offset'])
        if obj.fingerprint != m['fingerprint']:
            raise ValueError('Instance checksum mismatch')
        return obj

    @property
    def d(self):
        return self.Q.shape[0]

    @property
    def identifier(self):
        s = self.spec
        q = format(s.q, '.8g').replace('.', 'p').replace('-', 'm')
        return f'n{s.n}_T{s.T}_k{s.k}_B{s.B}_C{s.C}_i{s.instance_seed:03d}_q{q}'

    @property
    def position_bits(self):
        s = self.spec
        return s.T * 2 * s.n * s.k

    @property
    def fingerprint(self):
        h = hashlib.sha256()
        for a in [self.Q, self.economic_Q, self.A, self.b, self.up, self.covariance]:
            h.update(np.asarray(a, dtype='<f8').tobytes())
        h.update(json.dumps(asdict(self.spec), sort_keys=True).encode())
        h.update(json.dumps(self.assets).encode())
        return h.hexdigest()

    def indices(self, t):
        s = self.spec
        base = t * s.per_time
        nx = 2 * s.n * s.k
        x = np.arange(base, base + nx).reshape(s.n, s.k, 2)
        cash = np.arange(base + nx, base + nx + s.cash_bits)
        gross = np.arange(base + nx + s.cash_bits, base + s.per_time)
        return x, cash, gross

    def energies(self, x, economic=False):
        x = np.asarray(x, dtype=float)
        Q = self.economic_Q if economic else self.Q
        return np.einsum('...i,ij,...j->...', x, Q, x) + (0 if economic else self.offset)

    def feasible(self, x):
        return np.all(np.abs(np.asarray(x, dtype=float) @ self.A.T - self.b) < 1e-9, axis=-1)

    def encode_positions(self, positions):
        """Holdings (T, n, k, 2) -> full QUBO vector; the slack bits follow from A x = b."""
        s = self.spec
        positions = np.asarray(positions, dtype=int)
        out = np.zeros(self.d, dtype=np.int8)
        for t in range(s.T):
            xx, cc, gg = self.indices(t)
            z = positions[t]
            cash = s.C - int(z[..., 0].sum() - z[..., 1].sum())
            slack = s.B - int(z.sum())
            if not (0 <= cash <= s.cash_cap and 0 <= slack <= s.B):
                raise ValueError('Infeasible positions cannot be encoded')
            out[xx.ravel()] = z.ravel()
            out[cc] = [(cash >> b) & 1 for b in range(s.cash_bits)]
            out[gg] = [(slack >> b) & 1 for b in range(s.gross_bits)]
        return out

    def all_cash(self):
        s = self.spec
        return self.encode_positions(np.zeros((s.T, s.n, s.k, 2), dtype=int))

    def holdings(self, bits):
        """Position register (..., position_bits) -> holdings (..., T, n, k, 2)."""
        s = self.spec
        bits = np.asarray(bits)
        return bits.reshape(bits.shape[:-1] + (s.T, s.n, s.k, 2))

    def holding_objective(self, Z):
        """Economic objective and constraint violation of holdings Z (S, T, n, k, 2).
        For feasible holdings the objective equals the BQP objective."""
        s = self.spec
        Z = np.asarray(Z, dtype=float)
        net = (Z[..., 0] - Z[..., 1]).sum(axis=-1)
        exposure = self.up[:-1] * net
        risk = np.einsum('sti,tij,stj->s', exposure, self.covariance, exposure)
        pnl = ((self.up[1:] - self.up[:-1]) * net).sum(axis=(1, 2))
        short = s.rho_s * (self.up[:-1] * Z[..., 1].sum(axis=-1)).sum(axis=(1, 2))
        cash = s.rho_c * s.unit * (s.C - net.sum(axis=2)).sum(axis=1)
        prev = np.concatenate([np.zeros_like(Z[:, :1]), Z[:, :-1]], axis=1)
        fee = s.delta * (self.up[:-1, :, None, None] * np.abs(Z - prev)).sum(axis=(1, 2, 3, 4))
        fee += s.delta * (self.up[-1, :, None, None] * Z[:, -1]).sum(axis=(1, 2, 3))
        economic = s.q * risk - (pnl + cash - short - fee)
        total = net.sum(axis=2)
        violation = (np.maximum(0, Z.sum(axis=(2, 3, 4)) - s.B) + np.maximum(0, total - s.C)
                     + np.maximum(0, s.C - total - s.cash_cap)).sum(axis=1)
        return economic, violation


def bitstring_to_x(bitstring):
    """Qiskit bitstring (bit 0 rightmost) -> vector x[0], x[1], ..."""
    return np.array([int(c) for c in bitstring[::-1]], dtype=np.int8)


def x_to_bitstring(x):
    return ''.join(str(int(v)) for v in np.asarray(x)[::-1])
