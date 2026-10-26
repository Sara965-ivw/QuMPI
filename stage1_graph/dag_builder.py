"""Stage 1A: QASM -> QuantumCircuit -> DAG conversion.
Input contract:  path to an OpenQASM 2.0 file, or a QuantumCircuit.
Output contract: (QuantumCircuit, DAGCircuit) pair.
"""
from qiskit import QuantumCircuit, qasm2
from qiskit.converters import circuit_to_dag


def load_qasm(qasm_path: str) -> QuantumCircuit:
    return qasm2.load(qasm_path)


def to_dag(qc: QuantumCircuit):
    return circuit_to_dag(qc)


if __name__ == "__main__":
    # Self-test: 4-qubit GHZ state
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)

    dag = to_dag(qc)
    print("Circuit:")
    print(qc.draw(output="text"))
    print("\nTopological op order:")
    for node in dag.topological_op_nodes():
        qubits = [qc.find_bit(q).index for q in node.qargs]
        print(f"  {node.op.name:>4} on qubits {qubits}")
