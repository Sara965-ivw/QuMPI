"""Seeded circuit generators for the three Phase 1 families (Roadmap SS4.2).

Every generator takes an explicit integer seed and is deterministic given it
(numpy default_rng; no global state). Family membership, seed, and generator
parameters are returned as metadata for the dataset datasheet.
"""
from __future__ import annotations

import math

import numpy as np

from .circuits import Circuit


def random_circuit(n_qubits: int, depth: int, density: float, seed: int):
    """Unstructured family. `density` in (0,1] is the per-layer probability of
    entangling each available adjacent-in-permutation qubit pair — the knob
    that moves the w_ij landscape (Roadmap SS4.2)."""
    rng = np.random.default_rng(seed)
    circ = Circuit(n_qubits)
    for _ in range(depth):
        for q in range(n_qubits):
            gate = rng.choice(["rx", "ry", "rz", "h"])
            if gate == "h":
                circ.add("h", q)
            else:
                circ.add(str(gate), q, float(rng.uniform(0, 2 * math.pi)))
        perm = rng.permutation(n_qubits)
        for i in range(0, n_qubits - 1, 2):
            if rng.random() < density:
                circ.add("cx", (int(perm[i]), int(perm[i + 1])))
    meta = {"family": "random", "n_qubits": n_qubits, "depth": depth,
            "density": density, "seed": seed}
    return circ, meta


def twolocal_circuit(n_qubits: int, reps: int, seed: int,
                     entanglement: str = "linear"):
    """VQE-ansatz family: ry rotation layers + cx entangler, TwoLocal pattern."""
    rng = np.random.default_rng(seed)
    circ = Circuit(n_qubits)
    for _ in range(reps):
        for q in range(n_qubits):
            circ.add("ry", q, float(rng.uniform(0, 2 * math.pi)))
        pairs = ([(q, q + 1) for q in range(n_qubits - 1)] if entanglement == "linear"
                 else [(a, b) for a in range(n_qubits) for b in range(a + 1, n_qubits)])
        for a, b in pairs:
            circ.add("cx", (a, b))
    for q in range(n_qubits):  # final rotation layer
        circ.add("ry", q, float(rng.uniform(0, 2 * math.pi)))
    meta = {"family": "twolocal", "n_qubits": n_qubits, "reps": reps,
            "entanglement": entanglement, "seed": seed}
    return circ, meta


def qaoa_maxcut_circuit(n_qubits: int, p_layers: int, edge_prob: float, seed: int):
    """QAOA family on a random G(n, p) MaxCut instance; the problem-graph
    topology transfers into circuit topology (Roadmap SS4.2)."""
    rng = np.random.default_rng(seed)
    edges = [(a, b) for a in range(n_qubits) for b in range(a + 1, n_qubits)
             if rng.random() < edge_prob]
    if not edges:  # guarantee a connected-ish instance
        edges = [(q, q + 1) for q in range(n_qubits - 1)]
    circ = Circuit(n_qubits)
    for q in range(n_qubits):
        circ.add("h", q)
    for _ in range(p_layers):
        gamma = float(rng.uniform(0, math.pi))
        beta = float(rng.uniform(0, math.pi))
        for a, b in edges:
            circ.add("rzz", (a, b), gamma)
        for q in range(n_qubits):
            circ.add("rx", q, 2 * beta)
    meta = {"family": "qaoa_maxcut", "n_qubits": n_qubits, "p_layers": p_layers,
            "edge_prob": edge_prob, "n_edges": len(edges), "seed": seed}
    return circ, meta


# ------------------------------------------------- cuttable families (scheme v3)
def _blocks(n_qubits: int, n_blocks: int, rng) -> list:
    """Split qubits into contiguous blocks of near-equal size, randomly permuted
    so block membership is not encoded in qubit index order."""
    perm = rng.permutation(n_qubits).tolist()
    sizes = [n_qubits // n_blocks + (1 if i < n_qubits % n_blocks else 0) for i in range(n_blocks)]
    out, start = [], 0
    for s in sizes:
        out.append(sorted(perm[start:start + s]))
        start += s
    return out


def clustered_random_circuit(n_qubits: int, depth: int, n_blocks: int,
                             p_bridge: float, seed: int, density: float = 0.8):
    """Clustered random family (Peng et al. 2020 consider clustered circuits):
    dense random entangling layers inside each block, and in each layer at most
    one inter-block CX with probability p_bridge. Few inter-block gates is what
    makes a circuit worth cutting."""
    rng = np.random.default_rng(seed)
    blocks = _blocks(n_qubits, n_blocks, rng)
    circ = Circuit(n_qubits)
    n_bridges = 0
    for _ in range(depth):
        for q in range(n_qubits):
            g = rng.choice(["rx", "ry", "rz", "h"])
            if g == "h":
                circ.add("h", q)
            else:
                circ.add(str(g), q, float(rng.uniform(0, 2 * math.pi)))
        for blk in blocks:
            perm = rng.permutation(blk).tolist()
            for i in range(0, len(perm) - 1, 2):
                if rng.random() < density:
                    circ.add("cx", (int(perm[i]), int(perm[i + 1])))
        if n_blocks > 1 and rng.random() < p_bridge:
            b1, b2 = rng.choice(n_blocks, size=2, replace=False)
            circ.add("cx", (int(rng.choice(blocks[b1])), int(rng.choice(blocks[b2]))))
            n_bridges += 1
    meta = {"family": "clustered_random", "n_qubits": n_qubits, "depth": depth,
            "n_blocks": n_blocks, "p_bridge": p_bridge, "density": density,
            "n_bridges": n_bridges, "seed": seed}
    return circ, meta


def qaoa_sbm_circuit(n_qubits: int, p_layers: int, n_blocks: int, p_in: float,
                     n_inter: int, seed: int):
    """QAOA-MaxCut on a stochastic-block-model graph: dense communities joined
    by exactly n_inter inter-community edges (a cuttable problem topology)."""
    rng = np.random.default_rng(seed)
    blocks = _blocks(n_qubits, n_blocks, rng)
    edges = set()
    for blk in blocks:
        for i, a in enumerate(blk):
            for b in blk[i + 1:]:
                if rng.random() < p_in:
                    edges.add((min(a, b), max(a, b)))
        for a, b in zip(blk, blk[1:]):          # keep each community connected
            edges.add((min(a, b), max(a, b)))
    while sum(1 for a, b in edges
              if not any(a in blk and b in blk for blk in blocks)) < n_inter:
        b1, b2 = rng.choice(n_blocks, size=2, replace=False)
        a, b = int(rng.choice(blocks[b1])), int(rng.choice(blocks[b2]))
        edges.add((min(a, b), max(a, b)))
    edges = sorted(edges)
    circ = Circuit(n_qubits)
    for q in range(n_qubits):
        circ.add("h", q)
    for _ in range(p_layers):
        gamma = float(rng.uniform(0, math.pi))
        beta = float(rng.uniform(0, math.pi))
        for a, b in edges:
            circ.add("rzz", (a, b), gamma)
        for q in range(n_qubits):
            circ.add("rx", q, 2 * beta)
    meta = {"family": "qaoa_sbm", "n_qubits": n_qubits, "p_layers": p_layers,
            "n_blocks": n_blocks, "p_in": p_in, "n_inter": n_inter,
            "n_edges": len(edges), "seed": seed}
    return circ, meta
