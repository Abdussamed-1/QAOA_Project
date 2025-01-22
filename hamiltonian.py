from qiskit.quantum_info import SparsePauliOp
import numpy as np
from typing import List, Tuple

class HamiltonianBuilder:
    def __init__(
        self, num_qubits: int = None, edges: List[Tuple[int, int]] = None, weights: List[float] = None, weight_matrix: np.ndarray = None
    ):
        """Initialize with either edges and weights or a weight matrix.
        
        Args:
            num_qubits: Number of qubits in the system (optional if weight_matrix is provided)
            edges: List of tuples containing connected node indices (edges)
            weights: List of weights for each edge (must match edges)
            weight_matrix: Symmetric numpy array representing the weight matrix
        """
        if weight_matrix is not None:
            if weight_matrix.shape[0] != weight_matrix.shape[1]:
                raise ValueError("Weight matrix must be square.")
            if not np.allclose(weight_matrix, weight_matrix.T):
                raise ValueError("Weight matrix must be symmetric.")
            
            self.weight_matrix = weight_matrix
            self.num_qubits = weight_matrix.shape[0]
            self.edges = None
            self.weights = None
        else:
            if edges is None or weights is None:
                raise ValueError("Either weight_matrix or both edges and weights must be provided.")
            if len(edges) != len(weights):
                raise ValueError("Number of edges must match number of weights.")
            
            self.edges = edges
            self.weights = weights
            self.num_qubits = num_qubits
            self.weight_matrix = self._build_weight_matrix_from_edges()

    def _build_weight_matrix_from_edges(self) -> np.ndarray:
        """Create a weight matrix from edges and weights.
        
        Returns:
            A symmetric numpy array representing the weight matrix.
        """
        weight_matrix = np.zeros((self.num_qubits, self.num_qubits))
        for (node1, node2), weight in zip(self.edges, self.weights):
            weight_matrix[node1, node2] = weight
            weight_matrix[node2, node1] = weight  # Symmetric matrix
        return weight_matrix

    def build_pauli_list(self) -> List[Tuple[str, float]]:
        """Convert the graph to a Pauli list for Max-Cut.
        
        Returns:
            List of tuples containing Pauli strings and their coefficients
        """
        pauli_list = []
        # Extract ZZ terms from the weight matrix
        for i in range(self.num_qubits):
            for j in range(i + 1, self.num_qubits):  # Avoid double counting
                weight = self.weight_matrix[i, j]
                if weight != 0:  # Include only non-zero weights
                    paulis = ["I"] * self.num_qubits
                    paulis[i], paulis[j] = "Z", "Z"
                    pauli_list.append(("".join(paulis), -0.5 * weight))
        # Add single Z terms for each qubit
        for i in range(self.num_qubits):
            paulis = ["I"] * self.num_qubits
            paulis[i] = "Z"
            pauli_list.append(("".join(paulis), 0.5 * sum(self.weight_matrix[i])))
        return pauli_list

    def create_hamiltonian(self) -> SparsePauliOp:
        """Create the Hamiltonian operator.
        
        Returns:
            SparsePauliOp representing the Hamiltonian
        """
        pauli_list = self.build_pauli_list()
        cost_hamiltonian = SparsePauliOp.from_list(pauli_list)
        return cost_hamiltonian