from qiskit.quantum_info import SparsePauliOp
from typing import List, Tuple

class HamiltonianBuilder:
    def __init__(self, num_qubits: int, node_pairs: List[Tuple[int, int]], multipliers: List[float]):
        """Initialize with number of qubits and connection information
        
        Args:
            num_qubits: Number of qubits in the system
            node_pairs: List of tuples containing connected node indices
            multipliers: List of multiplier values for each connection
        """
        self.num_qubits = num_qubits
        self.node_pairs = node_pairs
        self.multipliers = multipliers
        
    def build_pauli_list(self) -> List[Tuple[str, float]]:
        """Convert the system to Pauli list based on node connections.
        
        Returns:
            List of tuples containing Pauli strings and their coefficients
        """
        pauli_list = []
        
        # Handle ZZ terms for each connection
        for (node1, node2), multiplier in zip(self.node_pairs, self.multipliers):
            paulis = ["I"] * self.num_qubits
            paulis[node1], paulis[node2] = "Z", "Z"
            pauli_list.append(("".join(paulis)[::1], multiplier))
            
        # Single Z terms for each node
        for node in range(self.num_qubits):
            paulis = ["I"] * self.num_qubits
            paulis[node] = "Z"
            pauli_list.append(("".join(paulis)[::1], -0.5))
        
        return pauli_list
    
    def create_hamiltonian(self) -> SparsePauliOp:
        """Create the Hamiltonian operator.
        
        Returns:
            SparsePauliOp representing the Hamiltonian
        """
        pauli_list = self.build_pauli_list()
        cost_hamiltonian = SparsePauliOp.from_list(pauli_list)
        return cost_hamiltonian
