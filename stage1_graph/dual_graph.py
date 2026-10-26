"""Stage 1B: DAG -> dual-graph representation.
Input contract:  (QuantumCircuit, DAGCircuit) from dag_builder.
Output contract: (GDG: nx.DiGraph, QIG: nx.Graph)
  GDG nodes: gates, with gate_type + qubit indices (+ hardware feature slots)
  GDG edges: temporal wire dependencies (directed)
  QIG nodes: physical qubits
  QIG edges: weighted by two-qubit interaction frequency  -> QUBO w_ij
"""
import networkx as nx
from qiskit.dagcircuit import DAGOpNode


def build_dual_graphs(qc, dag):
    # --- Gate Dependency Graph (directed, temporal) ---
    gdg = nx.DiGraph()
    node_index = {}  # DAGOpNode -> integer id, stable topological numbering
    for i, node in enumerate(dag.topological_op_nodes()):
        node_index[node] = i
        gdg.add_node(
            i,
            gate_type=node.op.name,
            qubits=tuple(qc.find_bit(q).index for q in node.qargs),
            # Hardware feature slots -- schema fixed now, populated in Phase 1
            t1=None, t2=None, gate_error=None,
        )
    for src, dst, _wire in dag.edges():
        if isinstance(src, DAGOpNode) and isinstance(dst, DAGOpNode):
            gdg.add_edge(node_index[src], node_index[dst])

    # --- Qubit Interaction Graph (undirected, weighted) ---
    qig = nx.Graph()
    qig.add_nodes_from(range(qc.num_qubits))
    for node in dag.topological_op_nodes():
        if len(node.qargs) == 2:
            q0, q1 = (qc.find_bit(q).index for q in node.qargs)
            if qig.has_edge(q0, q1):
                qig[q0][q1]["weight"] += 1
            else:
                qig.add_edge(q0, q1, weight=1)

    return gdg, qig


if __name__ == "__main__":
    from qiskit import QuantumCircuit
    from dag_builder import to_dag

    # Self-test circuit chosen so weights are hand-checkable:
    # qubits 0-1 interact THREE times, 1-2 twice, 2-3 once.
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(0, 1)
    qc.cx(2, 3)
    qc.cx(1, 2)
    qc.cx(0, 1)

    gdg, qig = build_dual_graphs(qc, to_dag(qc))

    print("GDG is a DAG (acyclic):", nx.is_directed_acyclic_graph(gdg))
    print(f"GDG: {gdg.number_of_nodes()} gates, {gdg.number_of_edges()} dependency edges")
    print("\nQIG edge weights / QUBO w_ij:")
    for u, v, d in sorted(qig.edges(data=True)):
        print(f"  w_{{{u},{v}}} = {d['weight']}")
