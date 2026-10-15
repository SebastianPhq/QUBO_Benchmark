from pathlib import Path
from .model import PortfolioQUBO

ROOT = Path(__file__).resolve().parent.parent


def instances(only=None):
    """(identifier, problem) for the eight Exp3 instances, ordered by q."""
    problems = [PortfolioQUBO.load(p) for p in sorted((ROOT/'instances').glob('*.npz'))]
    problems.sort(key=lambda p: p.spec.q)
    return [(p.identifier, p) for p in problems if not only or p.identifier in only]
