"""Bias-field warm-start QAOA for the portfolio BQP (Algorithm 1 of the paper).

Only the T*2nk position bits are qubits; the slack bits follow from A x = b. The
state prod_i Ry(theta_i)|0> with theta_i = 2 arcsin sqrt(pi_i) encodes the bias pi.
One QAOA layer follows: exp(-i gamma H_C) and the warm-start mixer
Ry(theta) Rz(2 beta) Ry(-theta). Linear terms are exact RZ rotations, the
rebalancing couplings between consecutive intervals are RZZ gates along chains,
and couplings that are not implemented as gates enter at their mean-field value.
After every round, each replica's bias moves toward the log-rank-weighted
marginals of its best samples (cross-entropy update).

mode='simulator' samples the circuit exactly (noiseless; every chain coupling is a
gate). mode='ibm' submits one SamplerV2 job per round to an IBM backend.
"""
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import gzip
import hashlib
import json
import numpy as np
from .model import bitstring_to_x, x_to_bitstring


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def versions():
    out = {}
    for name in ['numpy', 'scipy', 'qiskit', 'qiskit-ibm-runtime']:
        try: out[name] = version(name)
        except PackageNotFoundError: pass
    return out


def write_json(path, value, compress=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=None if compress else 2, default=str)
    temp = path.with_name(path.name + '.tmp')
    if compress:
        with gzip.open(temp, 'wt', compresslevel=1) as f: f.write(text)
    else:
        temp.write_text(text)
    temp.replace(path)


def read_json(path):
    path = Path(path)
    if path.suffix == '.gz':
        with gzip.open(path, 'rt') as f: return json.load(f)
    return json.loads(path.read_text())


def _angle(p):
    return 2 * np.arcsin(np.sqrt(np.clip(p, 0., 1.)))


class Bias:
    """Holding probabilities pi of one replica."""

    def __init__(self, n, p0):
        self.p = np.full(n, p0, dtype=float)

    def update(self, elite, weights, eta, clip):
        w = np.asarray(weights, dtype=float)
        self.p += eta * (w @ elite / w.sum() - self.p)
        np.clip(self.p, clip, 1 - clip, out=self.p)


def chain_structure(problem):
    """One chain per (asset, block, direction): its position bits in intervals 1..T."""
    s = problem.spec
    m = 2 * s.n * s.k
    return [[t * m + j for t in range(s.T)] for j in range(m)]


def position_qubo(problem):
    """Exact quadratic form E(z) = const + z^T Q z of the objective on position bits."""
    m = problem.position_bits
    E = lambda X: problem.holding_objective(problem.holdings(X))[0]
    e0 = E(np.zeros((1, m), np.int8))[0]
    ei = E(np.eye(m, dtype=np.int8)) - e0
    ii, jj = np.triu_indices(m, 1)
    Q = np.diag(ei)
    for lo in range(0, len(ii), 20000):
        a, b = ii[lo:lo + 20000], jj[lo:lo + 20000]
        X = np.zeros((len(a), m), np.int8); X[np.arange(len(a)), a] = 1; X[np.arange(len(a)), b] = 1
        Q[a, b] = E(X) - e0 - ei[a] - ei[b]
    return Q


class QAOACost:
    """Cost layer for a given set of gate-implemented links."""

    def __init__(self, problem, links, layers, dgamma, dbeta):
        self.Q = position_qubo(problem)
        self.links = [tuple(l) for l in links]
        native = np.zeros_like(self.Q, dtype=bool)
        for a, b in self.links: native[min(a, b), max(a, b)] = True
        self.J = np.array([self.Q[min(a, b), max(a, b)] for a, b in self.links])
        self.dropped = np.where(native, 0., np.triu(self.Q, 1))
        self.scale = float(np.abs(np.concatenate([np.diag(self.Q), self.J])).max()) or 1.
        self.gammas = [dgamma * (k + 1) / layers for k in range(layers)]
        self.betas = [dbeta * (1 - k / layers) for k in range(layers)]

    def angles(self, p):
        """Warm-start angles, RZ and RZZ angles, and mixer angles for exp(-i gamma E/scale)."""
        theta = _angle(p)
        lin = np.diag(self.Q) + self.dropped @ p + self.dropped.T @ p     # mean field
        half = np.zeros_like(lin)
        for (a, b), j in zip(self.links, self.J): half[a] += j / 2; half[b] += j / 2
        hz = [-(g / self.scale) * (lin + half) for g in self.gammas]
        jzz = [(g / self.scale) * self.J / 2 for g in self.gammas]
        return theta, hz, jzz, list(self.betas)


def make_circuit(n, links, layers):
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    theta = ParameterVector('theta', n); hz = ParameterVector('hz', n * layers)
    jzz = ParameterVector('jzz', len(links) * layers); beta = ParameterVector('beta', layers)
    qc = QuantumCircuit(n, n)
    for q in range(n): qc.ry(theta[q], q)
    for k in range(layers):
        for q in range(n): qc.rz(hz[k * n + q], q)
        for l, (a, b) in enumerate(links): qc.rzz(jzz[k * len(links) + l], a, b)
        for q in range(n):
            qc.ry(-theta[q], q); qc.rz(2 * beta[k], q); qc.ry(theta[q], q)
    qc.measure(range(n), range(n))
    return qc


def _segments(n, links):
    """Connected components (paths) of the link graph, as ordered qubit lists."""
    nbr = {q: [] for q in range(n)}
    for a, b in links: nbr[a].append(b); nbr[b].append(a)
    seen, out = set(), []
    for q in range(n):
        if q in seen or len(nbr[q]) > 1: continue
        path, prev = [q], None
        while True:
            seen.add(path[-1])
            nxt = [r for r in nbr[path[-1]] if r != prev and r not in seen]
            if not nxt: break
            prev = path[-1]; path.append(nxt[0])
        out.append(path)
    return out


def segment_distribution(segment, theta, hz, jzz_by_link, betas):
    """Exact output distribution of one chain of the circuit (statevector of T qubits)."""
    L = len(segment)
    ry = lambda a: np.array([[np.cos(a / 2), -np.sin(a / 2)], [np.sin(a / 2), np.cos(a / 2)]], dtype=complex)
    rz = lambda a: np.diag([np.exp(-1j * a / 2), np.exp(1j * a / 2)])

    def apply(psi, U, i):
        psi = psi.reshape(2**(L - i - 1), 2, 2**i)
        return np.einsum('ij,ajb->aib', U, psi).reshape(-1)
    bits = (np.arange(2**L)[:, None] >> np.arange(L)) & 1
    psi = np.zeros(2**L, complex); psi[0] = 1
    for i, q in enumerate(segment): psi = apply(psi, ry(theta[q]), i)
    for k, b in enumerate(betas):
        for i, q in enumerate(segment): psi = apply(psi, rz(hz[k][q]), i)
        for i in range(L - 1):
            z = 1 - 2 * bits[:, i] * 1.0; z2 = 1 - 2 * bits[:, i + 1] * 1.0
            psi = psi * np.exp(-1j * jzz_by_link[k][(segment[i], segment[i + 1])] / 2 * z * z2)
        for i, q in enumerate(segment): psi = apply(psi, ry(theta[q]) @ rz(2 * b) @ ry(-theta[q]), i)
    prob = np.abs(psi)**2
    return prob / prob.sum(), bits


def calibration(backend):
    """Readout and two-qubit gate errors from the backend target."""
    target = backend.target
    twoq = next(g for g in ('cz', 'ecr', 'cx') if g in target.operation_names)
    readout = {}
    for q in range(backend.num_qubits):
        props = target['measure'].get((q,))
        readout[q] = 1. if props is None or props.error is None else float(props.error)
    edges = {}
    for qargs, props in target[twoq].items():
        if props is None or props.error is None: continue
        key = tuple(sorted(qargs))
        edges[key] = min(edges.get(key, 1.), float(props.error))
    return twoq, readout, edges


def select_chain_layout(backend, chains, max_readout=0.05, max_twoq=0.05, restarts=300, seed=0):
    """Place each chain on a physical path of qubits and couplers with small errors.
    Chains that do not fit are split; a cut coupling is then taken at mean field."""
    twoq, readout, edges = calibration(backend)
    good = {q for q, e in readout.items() if e <= max_readout}
    adj = {q: set() for q in good}
    for (a, b), e in edges.items():
        if a in good and b in good and e <= max_twoq: adj[a].add(b); adj[b].add(a)
    n = sum(len(c) for c in chains)
    if len(good) < n:
        raise ValueError(f'Only {len(good)} qubits pass max_readout={max_readout}; need {n}')
    rng = np.random.default_rng(seed)

    def find_path(length, used):
        starts = sorted((q for q in good if q not in used), key=lambda q: (len(adj[q]), rng.random()))
        for s0 in starts:
            stack = [[s0]]
            while stack:
                path = stack.pop()
                if len(path) == length: return path
                nxt = [q for q in adj[path[-1]] if q not in used and q not in path]
                rng.shuffle(nxt)
                stack.extend(path + [q] for q in sorted(nxt, key=lambda q: len(adj[q]), reverse=True))
        return None

    best = None
    for _ in range(restarts):
        used, placement = set(), []
        pending = [list(chains[i]) for i in rng.permutation(len(chains))]
        for parts in (1, 2, 3, None):
            later = []
            for chain in pending:
                k = len(chain) if parts is None else max(1, -(-len(chain) // parts))
                pieces = [chain[i:i + k] for i in range(0, len(chain), k)] if parts else [[q] for q in chain]
                paths = []
                for piece in pieces:
                    path = find_path(len(piece), used | {q for pth in paths for q in pth})
                    if path is None: break
                    paths.append(path)
                if len(paths) == len(pieces):
                    for piece, path in zip(pieces, paths):
                        placement.append((piece, path)); used.update(path)
                else: later.append(chain)
            pending = later
            if not pending: break
        if pending: continue
        links = sum(len(piece) - 1 for piece, _ in placement)
        cost = sum(readout[q] for _, path in placement for q in path)
        if best is None or (links, -cost) > (best[0], -best[1]): best = (links, cost, placement)
        if links == sum(len(c) - 1 for c in chains): break
    if best is None: raise ValueError('Could not place all chain qubits on good qubits')
    layout = [None] * n; links = []
    for piece, path in best[2]:
        for q, p in zip(piece, path): layout[q] = p
        links += list(zip(piece, piece[1:]))
    return layout, links, {'two_qubit_gate': twoq, 'links_requested': sum(len(c) - 1 for c in chains),
                           'links_placed': len(links), 'readout_error': [readout[p] for p in layout],
                           'link_gate_error': [edges[tuple(sorted((layout[a], layout[b])))] for a, b in links]}


def compile_for_backend(problem, backend, cfg):
    from qiskit.transpiler import generate_preset_pass_manager
    layout, links, quality = select_chain_layout(backend, chain_structure(problem),
                                                 cfg['max_readout'], cfg['max_twoq'])
    circuit = make_circuit(problem.position_bits, links, cfg['layers'])
    pm = generate_preset_pass_manager(optimization_level=1, backend=backend,
                                      initial_layout=layout, seed_transpiler=cfg['seed'])
    isa = pm.run(circuit)
    n2 = sum(len(inst.qubits) == 2 for inst in isa.data)
    if n2 > 2 * len(links) * cfg['layers'] or isa.layout.final_index_layout()[:problem.position_bits] != layout:
        raise RuntimeError('Transpiler inserted routing or moved qubits; refusing this layout')
    return isa, {'backend_name': backend.name, 'links_logical': links, 'physical_layout': layout,
                 'layout_quality': quality, 'compiled_depth': isa.depth(), 'compiled_two_qubit_gates': n2,
                 'compiled_operations': dict(isa.count_ops())}


class Simulator:
    """Exact sampling of the noiseless circuit; every chain coupling is a gate."""

    def __init__(self, problem, directory, cfg):
        self.problem, self.directory, self.seed = problem, Path(directory), cfg['seed']
        links = [l for c in chain_structure(problem) for l in zip(c, c[1:])]
        self.cost = QAOACost(problem, links, cfg['layers'], cfg['dgamma'], cfg['dbeta'])
        self.segments = _segments(problem.position_bits, self.cost.links)

    def evaluate(self, biases, shots, label):
        path = self.directory / 'evaluations' / f'{label}.json.gz'
        if path.exists(): return read_json(path)
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(f'{self.seed}/{label}'.encode()).digest()[:8], 'little'))
        counts = []
        for bias in biases:
            theta, hz, jzz, betas = self.cost.angles(bias.p)
            by_link = [{} for _ in betas]
            for k in range(len(betas)):
                for (a, b), angle in zip(self.cost.links, jzz[k]): by_link[k][(a, b)] = by_link[k][(b, a)] = angle
            bits = np.zeros((shots, self.problem.position_bits), dtype=np.int8)
            for segment in self.segments:
                prob, basis = segment_distribution(segment, theta, hz, by_link, betas)
                bits[:, segment] = basis[rng.choice(len(prob), size=shots, p=prob)]
            counts.append(dict(Counter(x_to_bitstring(x) for x in bits)))
        record = {'label': label, 'shots': shots, 'counts': counts, 'completed_at_utc': utcnow()}
        write_json(path, record, compress=True)
        return record


class IBMHardware:
    """One SamplerV2 job per round; completed jobs are reused when a run is resumed."""

    def __init__(self, problem, directory, cfg, backend_name, token, instance):
        from qiskit import qpy
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
        self.directory, self.cfg = Path(directory), cfg
        self.service = QiskitRuntimeService(channel='ibm_quantum_platform', token=token, instance=instance)
        backend = self.service.backend(backend_name)
        qpy_path, meta_path = self.directory / 'circuit.qpy', self.directory / 'circuit_metadata.json'
        if qpy_path.exists():
            with qpy_path.open('rb') as f: self.circuit = qpy.load(f)[0]
            self.metadata = read_json(meta_path)
        else:
            self.circuit, self.metadata = compile_for_backend(problem, backend, cfg)
            with qpy_path.open('wb') as f: qpy.dump(self.circuit, f)
            write_json(meta_path, self.metadata)
            write_json(self.directory / 'backend_properties.json.gz', backend.properties().to_dict(), compress=True)
        self.cost = QAOACost(problem, self.metadata['links_logical'], cfg['layers'], cfg['dgamma'], cfg['dbeta'])
        self.sampler = SamplerV2(mode=backend, options={
            'max_execution_time': cfg['max_execution_time'],
            'dynamical_decoupling': {'enable': False},
            'twirling': {'enable_gates': False, 'enable_measure': False}})

    def bind(self, bias):
        theta, hz, jzz, betas = self.cost.angles(bias.p)
        source = {'theta': theta, 'hz': np.concatenate(hz), 'jzz': np.concatenate(jzz), 'beta': np.array(betas)}
        return np.array([source[p.vector.name][p.index] for p in self.circuit.parameters])

    def evaluate(self, biases, shots, label):
        path = self.directory / 'evaluations' / f'{label}.json.gz'
        record = read_json(path) if path.exists() else None
        if record and record['state'] == 'complete': return record
        if record and record.get('job_id'):
            job = self.service.job(record['job_id'])
        elif record:
            raise RuntimeError(f'Submission of {label} is uncertain; check IBM Quantum workloads before resuming')
        else:
            record = {'label': label, 'shots': shots, 'state': 'submitting', 'submitted_at_utc': utcnow()}
            write_json(path, record, compress=True)      # persist intent before the network call
            job = self.sampler.run([(self.circuit, self.bind(b)) for b in biases], shots=shots)
            record.update(job_id=job.job_id(), state='submitted')
            write_json(path, record, compress=True)
            print(f'{label}: job {job.job_id()}', flush=True)
        result = job.result(timeout=self.cfg['wait_timeout'])
        record.update(state='complete', counts=[dict(pub.data.c.get_counts()) for pub in result],
                      metrics=json.loads(json.dumps(job.metrics(), default=str)), completed_at_utc=utcnow())
        write_json(path, record, compress=True)
        return record


def evaluate_counts(problem, counts):
    """Bits, shot weights, economic objective, and constraint violation of one replica's counts."""
    keys = list(counts)
    bits = np.array([bitstring_to_x(k) for k in keys])
    weights = np.array([counts[k] for k in keys], dtype=float)
    economic, violation = problem.holding_objective(problem.holdings(bits))
    return keys, bits, weights, economic, violation


def select_elite(keys, bits, weights, economic, violation, archive, cfg):
    """Distinct samples of this round plus the replica's best feasible samples so far,
    ranked by (violation, objective); the best mu receive log-rank weights."""
    for key, b, e, v in zip(keys, bits, economic, violation):
        if v == 0 and key not in archive:
            archive[key] = (float(e), b)
    if len(archive) > cfg['archive']:
        for key in sorted(archive, key=lambda k: archive[k][0])[cfg['archive']:]: del archive[key]
    seen = set(keys)
    extra = [k for k in archive if k not in seen]
    all_bits = np.concatenate([bits, np.array([archive[k][1] for k in extra]).reshape(-1, bits.shape[1])])
    all_economic = np.concatenate([economic, [archive[k][0] for k in extra]])
    all_violation = np.concatenate([violation, np.zeros(len(extra))])
    order = np.lexsort((all_economic, all_violation))
    mu = max(1, min(len(order), int(np.ceil(cfg['elite_fraction'] * cfg['shots']))))
    return all_bits[order[:mu]], np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))


def run(problem, output, mode, cfg, backend_name=None, token=None, instance=None):
    """Run Algorithm 1: cfg['rounds'] rounds and one final sampling job, with
    cfg['chains'] replicas per job. Rerunning the same command resumes the run."""
    directory = Path(output)
    contract = {'mode': mode, 'backend': backend_name, 'config': cfg, 'instance_fingerprint': problem.fingerprint}
    manifest_path = directory / 'manifest.json'
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest['contract'] != contract:
            raise ValueError('An existing run with different settings is in this folder')
    else:
        manifest = {'contract': contract, 'started_at_utc': utcnow(), 'versions': versions()}
        write_json(manifest_path, manifest)
    engine = (Simulator(problem, directory, cfg) if mode == 'simulator' else
              IBMHardware(problem, directory, cfg, backend_name, token, instance))
    biases = [Bias(problem.position_bits, cfg['initial_probability']) for _ in range(cfg['chains'])]
    archives = [{} for _ in biases]
    best = (np.inf, None)
    for r in range(cfg['rounds']):
        label = f'round_{r:03d}'
        record = engine.evaluate(biases, cfg['shots'], label)
        for bias, archive, counts in zip(biases, archives, record['counts']):
            keys, bits, weights, economic, violation = evaluate_counts(problem, counts)
            if (violation == 0).any() and economic[violation == 0].min() < best[0] - 1e-9:
                best = (float(economic[violation == 0].min()), label)
            elite, elite_weights = select_elite(keys, bits, weights, economic, violation, archive, cfg)
            bias.update(elite, elite_weights, cfg['learning_rate'], cfg['clip'])
        print(f'{mode} round {r + 1}/{cfg["rounds"]}: best feasible objective {best[0]:.2f}', flush=True)
    record = engine.evaluate(biases, cfg['final_shots'], 'final')
    for counts in record['counts']:
        _, _, _, economic, violation = evaluate_counts(problem, counts)
        if (violation == 0).any() and economic[violation == 0].min() < best[0] - 1e-9:
            best = (float(economic[violation == 0].min()), 'final')
    result = {'mode': mode, 'jobs': cfg['rounds'] + 1,
              'shots': cfg['chains'] * (cfg['rounds'] * cfg['shots'] + cfg['final_shots']),
              'best_feasible_objective': best[0], 'best_first_found_in': best[1],
              'completed_at_utc': utcnow()}
    write_json(directory / 'result.json', result)
    return result
